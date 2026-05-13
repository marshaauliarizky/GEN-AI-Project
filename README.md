# GenAI Project Customer Analytics Dashboard

**DVD Rental Intelligence**  
President University Information System  
Team: Fatwa, Marsha, Syakira


## About

GenAI Project is a Customer Analytics Dashboard built on the **dvdrental** PostgreSQL database. This project combines data analytics, machine learning, and a generative AI assistant into one interactive web dashboard making it easy to explore customer behavior, rental trends, revenue insights, and segment analysis.


## Features

**Dashboard Tabs**
- Overview key metrics: total customers, revenue, rentals, avg spend per customer
- Behavior favorite genres, rental frequency, duration, and revenue correlation
- Loyalty & Segments RFM-based segmentation and spending tier breakdown
- Customer Detail individual customer lookup with full rental and payment history
- ML Predictions revenue forecasting and churn risk using Linear Regression & Random Forest

**AI Chatbot (FAMS AI)**
- Natural language assistant powered by Groq API
- Can execute dashboard actions via chat: change chart types, switch themes, navigate tabs, apply filters
- Scope is restricted to dvdrental data and this dashboard only
- Supports voice input

**Interactive Charts**
- Top N Customers by Spending (adjustable slider)
- Customer by Country
- Favorite Genres horizontal bar, vertical bar, pie, donut
- Scatter plot: rentals vs revenue
- Rental frequency and duration distribution
- Cumulative revenue over time

**Customization**
- Multiple themes (Dark, Ocean, Sunset, Purple, Pink, etc.)
- Per-chart color and chart type customization via AI chat
- Global filters: month range, spending segment, top N slider


## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | HTML, CSS, JavaScript, Plotly.js |
| Backend | Python, FastAPI |
| Database | PostgreSQL (dvdrental) |
| AI / LLM | Groq API |
| ML | Scikit-learn (Linear Regression, Random Forest) |

## How to Run

1. Clone the repository
```bash
git clone https://github.com/marshaauliarizky/GenAI-Project.git
cd GenAI-Project
```
2. Install dependencies
```bash
pip install -r requirements.txt
```
3. Make sure PostgreSQL is running with the dvdrental database on `localhost:5432`

4. Run the backend
```bash
python backend.py
```
5. Open `coba_connected.html` in browser, or via local server on `http://127.0.0.1:5500/coba_connected.html`

## File Structure

GenAI-Project/
├── backend.py              # FastAPI backend — data endpoints & ML models
├── coba_connected.html     # Main dashboard frontend
├── test_groq.py            # Groq API connection test
└── .gitignore

FAMS Team: Fatwa, Marsha, Syakira.
