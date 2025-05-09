import streamlit as st
import yfinance as yf
import pandas as pd
import time
import matplotlib.pyplot as plt
from streamlit_autorefresh import st_autorefresh
import requests
from bs4 import BeautifulSoup
import os
import altair as alt
import math
import numpy as np
import arch

# Add CSS to make sidebar buttons have equal size
button_css = """
<style>
.stSidebar .stButton button {
    width: 100%;
    height: 50px;
    margin-bottom: 10px;
    font-size: 16px;
    font-weight: bold;
    border-radius: 5px;
    background-color: #f0f0f0;
    border: 1px solid #ccc;
    text-align: left;
}
.stSidebar .stButton {
    width: 100%;
}
</style>
"""

# Update CSS to make the page visually cohesive with the rest of the website
custom_css = """
<style>
[data-testid="stAppViewContainer"] {
    background: #20407A !important;
    color: #ffffff !important;
    font-family: 'Arial', sans-serif;
}

[data-testid="stMarkdownContainer"] h3, label {
    color: #d3d3d3 !important;
    font-family: 'Arial', sans-serif;
    font-size: 16px;
}

.stButton button {
    color: #ffffff !important;
    background-color: #2a5298 !important;
    font-weight: bold !important;
    border-radius: 5px;
    padding: 10px 20px;
    font-family: 'Arial', sans-serif;
    font-size: 14px;
}

.stNumberInput input, .stTextInput input, .stSelectbox select {
    background-color: #ffffff !important;
    color: #000000 !important;
    border: 1px solid #ccc !important;
    border-radius: 5px !important;
    padding: 8px !important;
    font-family: 'Arial', sans-serif;
    font-size: 14px;
}

.stAltairChart {
    border: 1px solid #ccc !important;
    border-radius: 5px !important;
    padding: 10px !important;
    background-color: #ffffff !important;
}
</style>
"""

# Inject CSS for equal-sized buttons
st.markdown(button_css, unsafe_allow_html=True)

# Inject updated CSS for cohesive styling
st.markdown(custom_css, unsafe_allow_html=True)

# Initialize session state for page if not set
if 'page' not in st.session_state:
    st.session_state.page = 'home'

# Replace the radio buttons with buttons for navigation
st.sidebar.title("Navigation")
labels = ['Home', 'Stock Pricing', 'Option Pricing', 'Predicting Options', 'Visualizing Greeks', 'Injecting Market Sentiment']
pages = ['home', 'stock_pricing', 'option_pricing', 'predicting_options', 'visualizing_greeks', 'market_sentiment']

# Create buttons for each page in the sidebar
for i, label in enumerate(labels):
    if st.sidebar.button(label):
        st.session_state.page = pages[i]

# Render content based on selected page
import streamlit.components.v1 as components

if st.session_state.page == 'home':
    st.write("Welcome to the **Home** page!")

    # Add sliders for Black-Scholes parameters
    st.sidebar.header("Black-Scholes Parameters")
    stock_price = st.sidebar.slider("Stock Price (S)", min_value=1.0, max_value=500.0, value=100.0, step=1.0)
    strike_price = st.sidebar.slider("Strike Price (X)", min_value=1.0, max_value=500.0, value=100.0, step=1.0)
    time_to_maturity = st.sidebar.slider("Time to Maturity (T, in years)", min_value=0.01, max_value=5.0, value=1.0, step=0.01)
    risk_free_rate = st.sidebar.slider("Risk-Free Rate (r)", min_value=0.0, max_value=0.2, value=0.18, step=0.01)
    volatility = st.sidebar.slider("Volatility (σ)", min_value=0.01, max_value=1.0, value=0.2, step=0.01)

    # Black-Scholes formula implementation
    def black_scholes(option_type, S, X, T, r, v):
        d1 = (math.log(S / X) + (r + v**2 / 2) * T) / (v * math.sqrt(T))
        d2 = d1 - v * math.sqrt(T)
        if option_type == "call":
            return S * cnd(d1) - X * math.exp(-r * T) * cnd(d2)
        elif option_type == "put":
            return X * math.exp(-r * T) * cnd(-d2) - S * cnd(-d1)

    # Cumulative normal distribution function
    def cnd(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    # Generate data for 3D surface plot
    S_values = np.linspace(50, 150, 50)
    T_values = np.linspace(0.01, 5.0, 50)
    S_grid, T_grid = np.meshgrid(S_values, T_values)
    call_prices = np.array([
        black_scholes("call", S, strike_price, T, risk_free_rate, volatility)
        for S, T in zip(np.ravel(S_grid), np.ravel(T_grid))
    ]).reshape(S_grid.shape)

    # Plot 3D surface
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(S_grid, T_grid, call_prices, cmap="plasma", edgecolor="k")
    ax.set_title("Call Option Prices (Black-Scholes)", fontsize=16, color="white", pad=20)
    ax.set_xlabel("Stock Price (S)", fontsize=12, color="white", labelpad=10)
    ax.set_ylabel("Time to Maturity (T)", fontsize=12, color="white", labelpad=10)
    ax.set_zlabel("Option Price", fontsize=12, color="white", labelpad=10)
    ax.tick_params(colors="white")
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1, label="Option Price")
    fig.patch.set_facecolor("#20407A")
    ax.set_facecolor("#20407A")
    ax.grid(color="#444444", linestyle="--", linewidth=0.5)

    # Add rotation controls
    st.pyplot(fig)

    # Add textual explanation below the 3D graph
    st.write("## The Intuition Behind The Black-Scholes Equation")
    st.write(
        """
        The Black-Scholes equation is a cornerstone of modern financial theory. It provides a way to price options by using a replication approach. 
        The intuition is based on the idea that if you can replicate the cash flow of an asset with a strategy, then the price of the asset should equal the cost of executing the strategy.

        ### Key Concepts:
        1. **Replication**: If the strategy and the asset have the same cash flows, then a portfolio that is short the asset and long the strategy has no risk.
        2. **Arbitrage Pricing**: If the asset trades for a higher price than the cost to execute the strategy, you can short the asset and execute the strategy to capture the excess cash flow. This competition ensures the price aligns with replication costs.

        ### Example:
        Consider pricing a European-style call option with the following terms:
        - 1 year to maturity
        - Spot price = $100
        - Strike price = $125
        - Risk-free rate = 10%
        - Volatility = 30%

        Using the Black-Scholes formula, the call option is worth $7.20. It has a delta of 0.40 and a 29% chance of expiring in-the-money.

        ### Dynamic Hedging:
        The Black-Scholes model also introduces the concept of dynamic hedging, where the portfolio is continuously adjusted to remain risk-free.

        ### Final Observations:
        While the model is not taken literally for computing absolute values, it serves as a thermometer to gauge market-implied volatility and demonstrates the replication approach in arbitrage pricing techniques.
        """
    )

    # Add a dropdown below the graph to show the equation
    with st.expander("The Basics of the Equation"):
        st.write(
            """
            **C = N(d₁)Sₜ - N(d₂)Ke⁻ʳᵗ**

            where:
            - **C** = call option price
            - **N** = CDF of the normal distribution
            - **Sₜ** = spot price of an asset
            - **K** = strike price
            - **r** = risk-free interest rate
            - **t** = time to maturity
            - **σ** = volatility of the asset

            **d₁ = [ln(Sₜ / K) + (r + σ² / 2)t] / (σ√t)**

            **d₂ = d₁ - σ√t**
            """
        )

elif st.session_state.page == 'stock_pricing':
    st.title("Stock Pricing")

    # Input fields for company name, time range
    if 'company_name' not in st.session_state:
        st.session_state.company_name = ""
    if 'time_range' not in st.session_state:
        st.session_state.time_range = "1d"

    company_name = st.text_input("Enter the company ticker symbol (e.g., GOOGL for Google):", value=st.session_state.company_name)
    time_range = st.selectbox("Select the time range for the graph:", [ "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"], index=["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"].index(st.session_state.time_range))

    if company_name:
        st.session_state.company_name = company_name
        st.session_state.time_range = time_range

        # Button to fetch data
        if st.button("Fetch Data"):
            try:
                # Fetch stock data using yfinance
                stock_data = yf.Ticker(company_name)
                hist = stock_data.history(period=time_range)

                if not hist.empty:
                    # Prepare data for the graph
                    hist.reset_index(inplace=True)

                    # Create Vega-Lite chart
                    # Load Vega-Lite JSON specification
                    vega_lite_spec = {
                        "$schema": "https://vega.github.io/schema/vega-lite/v5.20.1.json",
                        "title": {
                            "text": f"Stock Prices for {company_name}",
                            "color": "white"
                        },
                        "width": 704,
                        "height": 400,
                        "autosize": {
                            "type": "fit",
                            "contains": "padding"
                        },
                        "padding": {"bottom": 20},
                        "data": {
                            "values": hist.to_dict(orient="records")
                        },
                        "mark": {
                            "type": "line",
                            "color": "#fff200",
                        },
                        "encoding": {
                            "tooltip": [
                                {"field": "Date", "type": "temporal"},
                                {"field": "Close", "type": "quantitative"}
                            ],
                            "x": {
                                "field": "Date",
                                "title": "Date",
                                "type": "temporal"
                            },
                            "y": {
                                "field": "Close",
                                "title": "Closing Price",
                                "type": "quantitative"
                            }
                        },
                        "config": {
                            "font": "sans-serif",
                            "background": "#20407A",
                            "view": {
                                "stroke": "transparent"
                            },
                            "title": {
                                "color": "white",
                                "fontSize": 18
                            },
                            "axis": {
                                "labelColor": "white",
                                "titleColor": "white",
                                "gridColor": "#ecec11",
                                "domainColor": "#ecec11",
                                "tickColor": "green"
                            },
                            "legend": {
                                "labelColor": "white",
                                "titleColor": "white"
                            },
                            "header": {
                                "labelColor": "white",
                                "titleColor": "white"
                            }
                        }
                    }

                    # Update the Vega-Lite JSON specification to use the desired line color
                    

                    # Display the graph using Vega-Lite JSON specification
                    st.vega_lite_chart(hist, spec=vega_lite_spec, use_container_width=True)

                    # The Vega-Lite chart is already displayed above, so this line is removed.
                else:
                    st.error("No data found for the given company ticker symbol. Please check the symbol and try again.")
            except Exception as e:
                if "404" in str(e):
                    st.error("The requested stock data could not be found. Please verify the ticker symbol.")
                else:
                    st.error(f"An unexpected error occurred: {e}")
elif st.session_state.page == 'option_pricing':
    st.title("Option Pricing")

    company_name = st.text_input("Enter the company ticker symbol (e.g., GOOGL for Google):")

    if company_name:
        try:
            # Fetch live option chain data using yfinance
            stock_data = yf.Ticker(company_name)
            options_dates = stock_data.options

            if options_dates:
                selected_date = st.selectbox("Select Expiration Date:", options_dates)
                option_chain = stock_data.option_chain(selected_date)

                # Combine calls and puts data
                calls_data = option_chain.calls
                puts_data = option_chain.puts

                calls_data["Type"] = "Call"
                puts_data["Type"] = "Put"

                options_data = pd.concat([calls_data, puts_data])

                # Create Vega-Lite chart for option prices vs. strike prices
                vega_lite_spec = {
                    "$schema": "https://vega.github.io/schema/vega-lite/v5.20.1.json",
                    "title": {
                        "text": f"Option Prices for {company_name} (Exp: {selected_date})",
                        "color": "white"
                    },
                    "width": 704,
                    "height": 400,
                    "autosize": {
                        "type": "fit",
                        "contains": "padding"
                    },
                    "padding": {"bottom": 20},
                    "data": {
                        "values": options_data.to_dict(orient="records")
                    },
                    "mark": {
                        "type": "line",
                        "point": True
                    },
                    "encoding": {
                        "tooltip": [
                            {"field": "strike", "type": "quantitative", "title": "Strike Price"},
                            {"field": "lastPrice", "type": "quantitative", "title": "Option Price"},
                            {"field": "Type", "type": "nominal", "title": "Option Type"}
                        ],
                        "x": {
                            "field": "strike",
                            "title": "Strike Price",
                            "type": "quantitative"
                        },
                        "y": {
                            "field": "lastPrice",
                            "title": "Option Price",
                            "type": "quantitative"
                        },
                        "color": {
                            "field": "Type",
                            "type": "nominal",
                            "title": "Option Type",
                            "scale": {"range": ["#1f77b4", "#ff7f0e"]}
                        }
                    },
                    "config": {
                        "font": "sans-serif",
                        "background": "#20407A",
                        "view": {
                            "stroke": "transparent"
                        },
                        "title": {
                            "color": "white",
                            "fontSize": 18
                        },
                        "axis": {
                            "labelColor": "white",
                            "titleColor": "white",
                            "gridColor": "#ecec11",
                            "domainColor": "#ecec11",
                            "tickColor": "green"
                        },
                        "legend": {
                            "labelColor": "white",
                            "titleColor": "white"
                        },
                        "header": {
                            "labelColor": "white",
                            "titleColor": "white"
                        }
                    }
                }

                # Display the graph using Vega-Lite JSON specification
                st.vega_lite_chart(options_data, spec=vega_lite_spec, use_container_width=True)
            else:
                st.error("No options data available for the selected company.")
        except Exception as e:
            st.error(f"An error occurred while fetching option data: {e}")
    st.write("Welcome to the **Option Pricing** page!")

elif st.session_state.page == 'predicting_options':
    st.title("Predicting Option Pricing")

    company_name = st.text_input("Enter the company ticker symbol (e.g., GOOGL for Google):")

    # Display current stock price above input boxes
    if company_name:
        try:
            stock_data = yf.Ticker(company_name)
            stock_price = stock_data.history(period="1d")["Close"].iloc[-1]
            st.subheader(f"Current Stock Price: ${stock_price:.2f}")

            # Input fields for user parameters
            option_type = st.selectbox("Select Option Type:", ["call", "put"], index=0)
            risk_free_rate = st.number_input("Enter the risk-free interest rate (as a decimal):", min_value=0.0, value=0.05, step=0.01)
            volatility = st.number_input("Enter the volatility (as a decimal):", min_value=0.0, value=0.2, step=0.01)
            time_to_maturity = st.number_input("Enter the time to maturity (in years):", min_value=0.01, value=1.0, step=0.01)

            strike_price = st.number_input("Enter the strike price:", min_value=0.0, step=1.0)

            if st.button("Calculate Option Price"):
                # Black-Scholes formula implementation
                def black_scholes(option_type, S, X, T, r, v):
                    if option_type not in ["call", "put"]:
                        return None

                    d1 = (math.log(S / X) + (r + v * v / 2) * T) / (v * math.sqrt(T))
                    d2 = d1 - v * math.sqrt(T)
                    if option_type == "call":
                        value = S * cnd(d1) - X * math.exp(-r * T) * cnd(d2)
                    else:
                        value = X * math.exp(-r * T) * cnd(-d2) - S * cnd(-d1)

                    return value

                # Cumulative normal distribution function
                def cnd(x):
                    a1, a2, a3, a4, a5 = 0.31938153, -0.356563782, 1.781477937, -1.821255978, 1.330274429

                    if x < 0.0:
                        return 1 - cnd(-x)
                    else:
                        k = 1.0 / (1.0 + 0.2316419 * x)
                        return 1.0 - math.exp(-x * x / 2.0) / math.sqrt(2 * math.pi) * k * (a1 + k * (a2 + k * (a3 + k * (a4 + k * a5))))

                # Calculate option price
                option_price = black_scholes(option_type, stock_price, strike_price, time_to_maturity, risk_free_rate, volatility)
                st.write(f"The predicted {option_type} option price is: ${option_price:.2f}")

                # Generate graph for option price vs. strike price using Vega-Lite
                strike_prices = [stock_price * (1 + i * 0.07) for i in range(-7, 8)]
                option_prices = [black_scholes(option_type, stock_price, sp, time_to_maturity, risk_free_rate, volatility) for sp in strike_prices]

                # Ensure all calculations for option prices and strike prices are valid
                graph_data = pd.DataFrame({
                    "Strike Price": strike_prices,
                    "Option Price": option_prices
                })

                # Replace invalid values with NaN and drop them
                graph_data = graph_data.replace([np.inf, -np.inf], np.nan).dropna()

                # Check if the DataFrame is empty after cleaning
                if graph_data.empty:
                    st.warning("No valid data available to display the chart.")
                else:
                    # Proceed with rendering the chart
                    vega_lite_spec = {
                        "$schema": "https://vega.github.io/schema/vega-lite/v5.20.1.json",
                        "title": {
                            "text": "Option Price vs. Strike Price",
                            "color": "white"
                        },
                        "width": 704,
                        "height": 400,
                        "autosize": {
                            "type": "fit",
                            "contains": "padding"
                        },
                        "padding": {"bottom": 20},
                        "data": {
                            "values": graph_data.to_dict(orient="records")
                        },
                        "mark": {
                            "type": "line",
                            "color": "#fff200",
                        },
                        "encoding": {
                            "tooltip": [
                                {"field": "Strike Price", "type": "quantitative"},
                                {"field": "Option Price", "type": "quantitative"}
                            ],
                            "x": {
                                "field": "Strike Price",
                                "title": "Strike Price",
                                "type": "quantitative"
                            },
                            "y": {
                                "field": "Option Price",
                                "title": "Option Price",
                                "type": "quantitative"
                            }
                        },
                        "config": {
                            "font": "sans-serif",
                            "background": "#20407A",
                            "view": {
                                "stroke": "transparent"
                            },
                            "title": {
                                "color": "white",
                                "fontSize": 18
                            },
                            "axis": {
                                "labelColor": "white",
                                "titleColor": "white",
                                "gridColor": "#ecec11",
                                "domainColor": "#ecec11",
                                "tickColor": "green"
                            },
                            "legend": {
                                "labelColor": "white",
                                "titleColor": "white"
                            },
                            "header": {
                                "labelColor": "white",
                                "titleColor": "white"
                            }
                        }
                    }

                    st.vega_lite_chart(graph_data, spec=vega_lite_spec, use_container_width=True)

                # Add Black-Scholes PDE and related equations to the Predicting Options page
                st.write("### Black-Scholes Partial Differential Equation")
                st.write(
                    """
                    The Black-Scholes PDE is given by:

                    **∂V/∂t + 0.5σ²S²∂²V/∂S² + rS∂V/∂S - rV = 0**

                    where:
                    - **V** = option price
                    - **S** = stock price
                    - **t** = time to maturity
                    - **σ** = volatility of the asset
                    - **r** = risk-free interest rate

                    This equation is the foundation for deriving the Black-Scholes formula for pricing options.
                    """
                )

                st.write("### Black-Scholes Equation")
                st.write(
                    """
                    **C = N(d₁)Sₜ - N(d₂)Ke⁻ʳᵗ**

                    where:
                    - **C** = call option price
                    - **N** = CDF of the normal distribution
                    - **Sₜ** = spot price of an asset
                    - **K** = strike price
                    - **r** = risk-free interest rate
                    - **t** = time to maturity
                    - **σ** = volatility of the asset

                    **d₁ = [ln(Sₜ / K) + (r + σ² / 2)t] / (σ√t)**

                    **d₂ = d₁ - σ√t**
                    """
                )

                st.write(f"### Solution for Call Option Price")
                st.write(f"The predicted call option price is: ${option_price:.2f}")
                st.write("### Comparison with Option Pricing Page")
                try:
                    # Use session state to store selected expiration date
                    if "selected_date" not in st.session_state:
                        # Find the closest expiration date to the current date + time to maturity
                        current_date = pd.Timestamp.now()
                        target_date = current_date + pd.Timedelta(days=int(time_to_maturity * 365))
                        expiration_dates = pd.to_datetime(stock_data.options)
                        closest_date = min(expiration_dates, key=lambda x: abs(x - target_date))
                        st.session_state.selected_date = closest_date.strftime('%Y-%m-%d')

                    selected_date = st.selectbox(
                        "Select Expiration Date:",
                        stock_data.options,
                        index=stock_data.options.index(st.session_state.selected_date),
                        key="expiration_date"
                    )

                    # Update session state with the selected date
                    st.session_state.selected_date = selected_date

                    option_chain = stock_data.option_chain(selected_date)

                    calls_data = option_chain.calls
                    puts_data = option_chain.puts

                    calls_data["Type"] = "Call"
                    puts_data["Type"] = "Put"

                    options_data = pd.concat([calls_data, puts_data])

                    real_data = options_data[options_data["Type"].str.lower() == option_type]

                    comparison_data = pd.DataFrame({
                        "Strike Price": real_data["strike"],
                        "Option Price": real_data["lastPrice"],
                        "Type": ["Real Data"] * len(real_data)
                    })

                    combined_data = pd.concat([
                        pd.DataFrame({
                            "Strike Price": strike_prices,
                            "Option Price": option_prices,
                            "Type": ["Predicted"] * len(strike_prices)
                        }),
                        comparison_data
                    ])

                    comparison_spec = {
                        "$schema": "https://vega.github.io/schema/vega-lite/v5.20.1.json",
                        "title": {
                            "text": "Comparison of Predicted and Real Option Prices",
                            "color": "white"
                        },
                        "width": 704,
                        "height": 400,
                        "autosize": {
                            "type": "fit",
                            "contains": "padding"
                        },
                        "padding": {"bottom": 20},
                        "data": {
                            "values": combined_data.to_dict(orient="records")
                        },
                        "mark": {
                            "type": "line",
                            "point": True
                        },
                        "encoding": {
                            "tooltip": [
                                {"field": "Strike Price", "type": "quantitative"},
                                {"field": "Option Price", "type": "quantitative"},
                                {"field": "Type", "type": "nominal"}
                            ],
                            "x": {
                                "field": "Strike Price",
                                "title": "Strike Price",
                                "type": "quantitative"
                            },
                            "y": {
                                "field": "Option Price",
                                "title": "Option Price",
                                "type": "quantitative"
                            },
                            "color": {
                                "field": "Type",
                                "type": "nominal",
                                "title": "Data Type",
                                "scale": {"range": ["#1f77b4", "#ff7f0e"]}
                            }
                        },
                        "config": {
                            "font": "sans-serif",
                            "background": "#20407A",
                            "view": {
                                "stroke": "transparent"
                            },
                            "title": {
                                "color": "white",
                                "fontSize": 18
                            },
                            "axis": {
                                "labelColor": "white",
                                "titleColor": "white",
                                "gridColor": "#ecec11",
                                "domainColor": "#ecec11",
                                "tickColor": "green"
                            },
                            "legend": {
                                "labelColor": "white",
                                "titleColor": "white"
                            },
                            "header": {
                                "labelColor": "white",
                                "titleColor": "white"
                            }
                        }
                    }

                    st.vega_lite_chart(combined_data, spec=comparison_spec, use_container_width=True)
                except Exception as e:
                    st.error(f"An error occurred while fetching real option data for comparison: {e}")

        except Exception as e:
            st.error(f"An error occurred while fetching stock data or calculating the option price: {e}")
elif st.session_state.page == 'visualizing_greeks':
    st.title("Visualizing Greeks")
    # Task 1: Visualize Delta, Vega, Gamma for different parameter sets
    st.header("Greeks for Different Parameter Sets")
    parameter_sets = [
        {"stock_price": 100, "strike_price": 100, "volatility": 0.2, "time_to_maturity": 1, "risk_free_rate": 0.05},
        {"stock_price": 110, "strike_price": 100, "volatility": 0.25, "time_to_maturity": 1, "risk_free_rate": 0.05},
        {"stock_price": 90, "strike_price": 100, "volatility": 0.2, "time_to_maturity": 0.5, "risk_free_rate": 0.05},
        {"stock_price": 100, "strike_price": 100, "volatility": 0.3, "time_to_maturity": 1, "risk_free_rate": 0.05},
    ]

    greeks_data = []
    for params in parameter_sets:
        stock_price = params["stock_price"]
        strike_price = params["strike_price"]
        volatility = params["volatility"]
        time_to_maturity = params["time_to_maturity"]
        risk_free_rate = params["risk_free_rate"]

        delta = (math.log(stock_price / strike_price) + (risk_free_rate + volatility**2 / 2) * time_to_maturity) / (volatility * math.sqrt(time_to_maturity))
        vega = stock_price * math.sqrt(time_to_maturity) * math.exp(-0.5 * delta**2) / math.sqrt(2 * math.pi)
        gamma = vega / (stock_price * volatility * math.sqrt(time_to_maturity))

        greeks_data.append({"Parameter Set": str(params), "Delta": delta, "Vega": vega, "Gamma": gamma})

    greeks_df = pd.DataFrame(greeks_data)

    for greek in ["Delta", "Vega", "Gamma"]:
        chart = alt.Chart(greeks_df).mark_line(point=True).encode(
            x=alt.X("Parameter Set", title="Parameter Set"),
            y=alt.Y(greek, title=greek),
            tooltip=["Parameter Set", greek]
        ).properties(
            title={"text": f"{greek} for Different Parameter Sets", "color": "white"},
            width=704,
            height=400
        ).configure(
            background="#20407a",
            font="sans-serif",
            title={"color": "white", "fontSize": 18},
            axis={
                "labelColor": "white",
                "titleColor": "white",
                "gridColor": "#ecec11",
                "domainColor": "#ecec11",
                "tickColor": "green"
            },
            legend={"labelColor": "white", "titleColor": "white"}
        )

        st.altair_chart(chart, use_container_width=True)

    # Task 2: Visualize Greeks for live stock data
    st.header("Greeks for Live Stock Data")
    company_name = st.text_input("Enter the company ticker symbol (e.g., GOOGL for Google):")

    if company_name:
        try:
            stock_data = yf.Ticker(company_name)
            stock_price = stock_data.history(period="1d")["Close"].iloc[-1]
            st.subheader(f"Current Stock Price: ${stock_price:.2f}")

            # Calculate Greeks for live data
            delta = (math.log(stock_price / 100) + (0.05 + 0.2**2 / 2) * 1) / (0.2 * math.sqrt(1))
            vega = stock_price * math.sqrt(1) * math.exp(-0.5 * delta**2) / math.sqrt(2 * math.pi)
            gamma = vega / (stock_price * 0.2 * math.sqrt(1))

            live_greeks_data = pd.DataFrame({
                "Greek": ["Delta", "Vega", "Gamma"],
                "Value": [delta, vega, gamma]
            })

            live_chart = alt.Chart(live_greeks_data).mark_bar().encode(
                x=alt.X("Greek", title="Greek"),
                y=alt.Y("Value", title="Value"),
                tooltip=["Greek", "Value"]
            ).properties(
                title={"text": "Greeks for Live Stock Data", "color": "white"},
                width=704,
                height=400
            ).configure(
                background="#20407a",
                font="sans-serif",
                title={"color": "white", "fontSize": 18},
                axis={
                    "labelColor": "white",
                    "titleColor": "white",
                    "gridColor": "#ecec11",
                    "domainColor": "#ecec11",
                    "tickColor": "green"
                },
                legend={"labelColor": "white", "titleColor": "white"}
            )

            st.altair_chart(live_chart, use_container_width=True)

        except Exception as e:
            st.error(f"An error occurred while fetching live stock data: {e}")

    # Task 3: User input for Greeks visualization
    st.header("User Input for Greeks Visualization")
    stock_price = st.number_input("Enter the stock price:", min_value=0.0, value=100.0, step=1.0)
    strike_price = st.number_input("Enter the strike price:", min_value=0.0, value=100.0, step=1.0)
    risk_free_rate = st.number_input("Enter the risk-free interest rate (as a decimal):", min_value=0.0, value=0.05, step=0.01)
    volatility = st.number_input("Enter the volatility (as a decimal):", min_value=0.0, value=0.2, step=0.01)
    time_to_maturity = st.number_input("Enter the time to maturity (in years):", min_value=0.01, value=1.0, step=0.01)

    if st.button("Visualize Greeks"):
        try:
            delta = (math.log(stock_price / strike_price) + (risk_free_rate + volatility**2 / 2) * time_to_maturity) / (volatility * math.sqrt(time_to_maturity))
            vega = stock_price * math.sqrt(time_to_maturity) * math.exp(-0.5 * delta**2) / math.sqrt(2 * math.pi)
            gamma = vega / (stock_price * volatility * math.sqrt(time_to_maturity))

            user_greeks_data = pd.DataFrame({
                "Greek": ["Delta", "Vega", "Gamma"],
                "Value": [delta, vega, gamma]
            })

            user_chart = alt.Chart(user_greeks_data).mark_bar().encode(
                x=alt.X("Greek", title="Greek"),
                y=alt.Y("Value", title="Value"),
                tooltip=["Greek", "Value"]
            ).properties(
                title={"text": "User Input Greeks Visualization", "color": "white"},
                width=704,
                height=400
            ).configure(
                background="#20407a",
                font="sans-serif",
                title={"color": "white", "fontSize": 18},
                axis={
                    "labelColor": "white",
                    "titleColor": "white",
                    "gridColor": "#ecec11",
                    "domainColor": "#ecec11",
                    "tickColor": "green"
                },
                legend={"labelColor": "white", "titleColor": "white"}
            )

            st.altair_chart(user_chart, use_container_width=True)

        except Exception as e:
            st.error(f"An error occurred while visualizing Greeks: {e}")
            st.error(f"An error occurred while visualizing Greeks: {e}")
elif st.session_state.page == 'market_sentiment':
    st.title("Injecting Market Sentiment")

    st.write("## Predicting Volatility Using GARCH Model")

    # User input for stock ticker and date range
    st.write("Enter the stock ticker and date range to fetch historical data:")
    stock_ticker = st.text_input("Stock Ticker (e.g., AAPL, GOOGL):")
    start_date = st.date_input("Start Date:")
    end_date = st.date_input("End Date:")

    if stock_ticker and start_date and end_date:
        try:
            # Fetch historical data from Yahoo Finance
            st.write("Fetching historical data...")
            stock_data = yf.download(stock_ticker, start=start_date, end=end_date, auto_adjust=False)

            if stock_data.empty:
                st.error("No data found for the given stock ticker and date range. Please check the ticker or adjust the date range.")
            else:
                # Calculate daily returns
                stock_data['Return'] = np.log(stock_data['Close'] / stock_data['Close'].shift(1))
                stock_data.dropna(inplace=True)

                # Rescale returns for GARCH model
                stock_data['Return'] *= 100

                # Fit GARCH(1,1) model
                st.write("Fitting GARCH(1,1) model...")
                from arch import arch_model
                model = arch_model(stock_data['Return'], vol='Garch', p=1, q=1, rescale=False)
                garch_fit = model.fit(disp='off')

                # Predict volatility for the next day
                forecast = garch_fit.forecast(horizon=1)
                predicted_volatility = np.sqrt(forecast.variance.iloc[-1, 0]) / 100  # Rescale back

                st.write(f"Predicted Volatility for the next day: {predicted_volatility:.4f}")

                # Ensure stock_price and predicted_volatility are floats
                stock_price = float(stock_data['Close'].iloc[-1])
                predicted_volatility = float(predicted_volatility)

                # Black-Scholes formula implementation
                def black_scholes(option_type, S, X, T, r, v):
                    d1 = (math.log(S / X) + (r + v**2 / 2) * T) / (v * math.sqrt(T))
                    d2 = d1 - v * math.sqrt(T)
                    if option_type == "call":
                        return S * cnd(d1) - X * math.exp(-r * T) * cnd(d2)
                    elif option_type == "put":
                        return X * math.exp(-r * T) * cnd(-d2) - S * cnd(-d1)

                # Cumulative normal distribution function
                def cnd(x):
                    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

                # Calculate call and put option prices
                # Take user inputs for strike price, time to maturity, and risk-free rate
                strike_price = st.number_input("Enter the strike price:", min_value=0.0, step=1.0)
                time_to_maturity = st.number_input("Enter the time to maturity (in years):", min_value=0.01, value=1.0, step=0.01)
                risk_free_rate = st.number_input("Enter the risk-free interest rate (as a decimal):", min_value=0.0, value=0.05, step=0.01)

                # Calculate call and put option prices
                call_price = black_scholes("call", stock_price, strike_price, time_to_maturity, risk_free_rate, predicted_volatility)
                put_price = black_scholes("put", stock_price, strike_price, time_to_maturity, risk_free_rate, predicted_volatility)

                st.write(f"The calculated call option price is: ${call_price:.2f}")
                st.write(f"The calculated put option price is: ${put_price:.2f}")

        except Exception as e:
            st.error(f"An error occurred while fetching data or fitting the model: {e}")
    else:
        st.write("Please enter the stock ticker and date range to proceed.")
else:
    st.write("Welcome to the **Stocks and Options** dashboard!")
