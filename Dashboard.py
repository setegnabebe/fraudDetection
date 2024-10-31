from flask import Flask, jsonify
import pandas as pd
import dash
from dash import dcc, html
import plotly.express as px

# Initialize Flask app
app = Flask(__name__)

# Load the fraud data CSV file once and cache it
fraud_data = pd.read_csv('data/Fraud_Data.csv')

# Load the IP address mapping data
ip_data = pd.read_csv('data/IpAddress_to_Country.csv')  # Update the path as needed

# Preprocess IP data to allow efficient lookups
ip_data['lower_bound_ip_address'] = ip_data['lower_bound_ip_address'].astype(int)
ip_data['upper_bound_ip_address'] = ip_data['upper_bound_ip_address'].astype(int)

# Function to map IP address to country
def map_ip_to_country(ip_address):
    ip_int = int(ip_address)
    country = ip_data[
        (ip_data['lower_bound_ip_address'] <= ip_int) & 
        (ip_data['upper_bound_ip_address'] >= ip_int)
    ]['country']
    
    return country.iloc[0] if not country.empty else 'Unknown'

# Map IP addresses to countries and cache the results
fraud_data['country'] = fraud_data['ip_address'].apply(map_ip_to_country)

@app.route('/api/fraud-data', methods=['GET'])
def fraud_data_api():
    try:
        total_transactions = len(fraud_data)
        total_frauds = fraud_data[fraud_data['class'] == 1].shape[0] if 'class' in fraud_data.columns else 0
        fraud_percentage = (total_frauds / total_transactions) * 100 if total_transactions > 0 else 0
        
        summary = {
            "total_transactions": total_transactions,
            "total_frauds": total_frauds,
            "fraud_percentage": fraud_percentage,
        }
        
        return jsonify(summary)
    except Exception as e:
        print(f"Error in fraud_data route: {e}")
        return jsonify({"error": "Internal Server Error"}), 500

# Initialize Dash app
app_dash = dash.Dash(__name__, server=app, url_base_pathname='/dashboard/')

# Layout of the dashboard
app_dash.layout = html.Div([
    html.H1("Fraud Detection Dashboard"),
    html.Div(id='summary-stats', style={'padding': '10px'}),
    dcc.Graph(id='line-chart'),
    html.Div(id='additional-info', style={'padding': '10px'}),
    dcc.Graph(id='location-map'),  # New graph for fraud by location
    dcc.Graph(id='globe-chart'),  # Globe chart for geographical fraud display
])

# Callback to update summary stats and graphs
@app_dash.callback(
    [dash.dependencies.Output('summary-stats', 'children'),
     dash.dependencies.Output('line-chart', 'figure'),
     dash.dependencies.Output('additional-info', 'children'),
     dash.dependencies.Output('location-map', 'figure'),  # Map for country fraud
     dash.dependencies.Output('globe-chart', 'figure')],  # New output for the globe chart
    [dash.dependencies.Input('summary-stats', 'children')]  # Dummy input to trigger the callback
)
def update_dashboard(_):
    # Get summary data
    total_transactions = len(fraud_data)
    total_frauds = fraud_data[fraud_data['class'] == 1].shape[0] if 'class' in fraud_data.columns else 0
    fraud_percentage = (total_frauds / total_transactions) * 100 if total_transactions > 0 else 0

    summary = [
        html.Div(f"Total Transactions: {total_transactions}"),
        html.Div(f"Total Fraud Cases: {total_frauds}"),
        html.Div(f"Fraud Percentage: {fraud_percentage:.2f}%")
    ]

    # Create the first line chart
    figure1 = px.line(fraud_data, x='purchase_time', y='purchase_value', title='Fraud Cases Over Time')

    # Create a map for fraudulent transactions by country
    figure2 = px.histogram(fraud_data[fraud_data['class'] == 1],  
                            x='country',
                            title='Fraudulent Transactions by Country',
                            color='country',
                            text_auto=True)

    # Display additional information
    additional_info = html.Div([
        html.H3("Additional Information"),
        html.Div(f"Sample User ID: {fraud_data['user_id'].iloc[0]}"),
        html.Div(f"Signup Time: {fraud_data['signup_time'].iloc[0]}"),
        html.Div(f"Device ID: {fraud_data['device_id'].iloc[0]}"),
        html.Div(f"Source: {fraud_data['source'].iloc[0]}"),
        html.Div(f"Browser: {fraud_data['browser'].iloc[0]}"),
        html.Div(f"Sex: {fraud_data['sex'].iloc[0]}"),
        html.Div(f"Age: {fraud_data['age'].iloc[0]}"),
    ])

    # Create a globe chart for geographical distribution of fraud
    globe_data = fraud_data[fraud_data['class'] == 1].groupby('country').size().reset_index(name='counts')
    figure3 = px.choropleth(globe_data,
                             locations='country',
                             locationmode='country names',
                             color='counts',
                             title='Geographical Distribution of Fraudulent Transactions',
                             color_continuous_scale=px.colors.sequential.Plasma)

    return summary, figure1, additional_info, figure2, figure3  # Return the globe chart

# Run the app
if __name__ == '__main__':
    app.run(port=5001, debug=True)
