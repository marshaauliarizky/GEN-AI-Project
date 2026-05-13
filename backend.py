# ══════════════════════════════════════════════════════════════
# backend_v2.py — FastAPI DVD Rental Analytics + ML Predictions
# ══════════════════════════════════════════════════════════════
# Install dependencies:
#   pip install fastapi uvicorn sqlalchemy psycopg2-binary pandas scikit-learn
#
# Jalankan:
#   uvicorn backend_v2:app --reload --port 8000
#
# Test:
#   http://localhost:8000/docs
# ══════════════════════════════════════════════════════════════

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, text
import pandas as pd
import numpy as np
import traceback

# ── ML imports ──
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────────────────
# ⚙️  KONFIGURASI DB
# ──────────────────────────────────────────────────────────────
DB_USER     = "postgres"
DB_PASSWORD = "marshacaca123"   # ← sesuaikan password kamu
DB_HOST     = "localhost"
DB_PORT     = "5432"
DB_NAME     = "rentalDB"

DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

engine = create_engine(DATABASE_URL, pool_pre_ping=True)

app = FastAPI(title="DVD Rental Analytics API v2", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────────────────────
# HELPER
# ──────────────────────────────────────────────────────────────
def run_query(sql: str, params: dict = None):
    try:
        with engine.connect() as conn:
            result = conn.execute(text(sql), params or {})
            rows   = result.fetchall()
            cols   = result.keys()
            return [dict(zip(cols, row)) for row in rows]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ══════════════════════════════════════════════════════════════
# ENDPOINTS DATA (sama dengan sebelumnya)
# ══════════════════════════════════════════════════════════════

@app.get("/")
def root():
    return {"status": "ok", "message": "DVD Rental Analytics API v2 running"}

@app.get("/api/ping")
def ping():
    try:
        run_query("SELECT 1")
        return {"status": "connected", "db": DB_NAME}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e}")

@app.get("/api/kpi")
def get_kpi():
    return run_query("""
        SELECT
            COUNT(DISTINCT c.customer_id)                                  AS total_customers,
            ROUND(SUM(p.amount)::numeric, 2)                               AS total_revenue,
            COUNT(DISTINCT r.rental_id)                                    AS total_rentals,
            ROUND(AVG(p.amount)::numeric, 2)                               AS avg_transaction,
            ROUND(SUM(p.amount) / COUNT(DISTINCT c.customer_id), 2)        AS avg_spend_per_customer
        FROM customer c
        JOIN rental  r ON c.customer_id = r.customer_id
        JOIN payment p ON r.rental_id   = p.rental_id
    """)

@app.get("/api/monthly")
def get_monthly():
    return run_query("""
        SELECT
            TO_CHAR(DATE_TRUNC('month', payment_date), 'YYYY-MM')  AS month,
            TO_CHAR(DATE_TRUNC('month', payment_date), 'Mon YYYY') AS label,
            ROUND(SUM(amount)::numeric, 2)                          AS revenue,
            COUNT(*)                                                AS transactions,
            COUNT(DISTINCT customer_id)                             AS active_customers
        FROM payment
        GROUP BY 1, 2
        ORDER BY 1
    """)

@app.get("/api/customers")
def get_customers():
    return run_query("""
        SELECT
            cu.customer_id,
            cu.first_name || ' ' || cu.last_name                        AS name,
            cu.email,
            co.country,
            cu.active,
            COUNT(DISTINCT r.rental_id)                                  AS rental_count,
            ROUND(SUM(p.amount)::numeric, 2)                             AS total_revenue,
            ROUND(AVG(p.amount)::numeric, 2)                             AS avg_payment,
            ROUND(AVG(
                EXTRACT(EPOCH FROM (r.return_date - r.rental_date)) / 86400.0
            )::numeric, 1)                                               AS avg_duration
        FROM customer cu
        JOIN address a  ON cu.address_id  = a.address_id
        JOIN city    ci ON a.city_id      = ci.city_id
        JOIN country co ON ci.country_id  = co.country_id
        JOIN rental  r  ON cu.customer_id = r.customer_id
        JOIN payment p  ON r.rental_id    = p.rental_id
        WHERE r.return_date IS NOT NULL
        GROUP BY cu.customer_id, name, cu.email, co.country, cu.active
        ORDER BY total_revenue DESC
    """)

@app.get("/api/rfm")
def get_rfm():
    return run_query("""
        WITH rfm AS (
            SELECT
                c.customer_id,
                c.first_name || ' ' || c.last_name                AS name,
                c.email,
                MAX(r.rental_date)::date                           AS last_rental,
                COUNT(DISTINCT r.rental_id)                        AS frequency,
                ROUND(SUM(p.amount)::numeric, 2)                   AS monetary,
                (SELECT MAX(rental_date)::date FROM rental)
                    - MAX(r.rental_date)::date                     AS recency_days
            FROM customer c
            JOIN rental  r ON c.customer_id = r.customer_id
            JOIN payment p ON r.rental_id   = p.rental_id
            GROUP BY c.customer_id, name, c.email
        )
        SELECT *,
            CASE
                WHEN recency_days <= 30  AND frequency >= 30 AND monetary >= 150 THEN 'Champions'
                WHEN recency_days <= 60  AND (frequency >= 20 OR monetary >= 100) THEN 'Loyal'
                WHEN recency_days BETWEEN 61 AND 120                              THEN 'At Risk'
                ELSE 'Lost'
            END AS segment
        FROM rfm
        ORDER BY monetary DESC
    """)

@app.get("/api/genre")
def get_genre():
    return run_query("""
        SELECT
            cat.name                        AS genre,
            COUNT(DISTINCT r.rental_id)     AS rental_count
        FROM rental r
        JOIN inventory     i   ON r.inventory_id  = i.inventory_id
        JOIN film          f   ON i.film_id        = f.film_id
        JOIN film_category fc  ON f.film_id        = fc.film_id
        JOIN category      cat ON fc.category_id   = cat.category_id
        GROUP BY cat.name
        ORDER BY rental_count DESC
    """)

# ── NEW: Genre per bulan (untuk AI bisa jawab "genre apa paling banyak bulan X") ──
@app.get("/api/genre-monthly")
def get_genre_monthly():
    """Rental count per genre per bulan. AI bisa query ini untuk jawab pertanyaan spesifik."""
    return run_query("""
        SELECT
            TO_CHAR(DATE_TRUNC('month', r.rental_date), 'YYYY-MM')  AS month,
            TO_CHAR(DATE_TRUNC('month', r.rental_date), 'Mon YYYY') AS label,
            cat.name                                                  AS genre,
            COUNT(DISTINCT r.rental_id)                               AS rental_count
        FROM rental r
        JOIN inventory     i   ON r.inventory_id  = i.inventory_id
        JOIN film          f   ON i.film_id        = f.film_id
        JOIN film_category fc  ON f.film_id        = fc.film_id
        JOIN category      cat ON fc.category_id   = cat.category_id
        GROUP BY 1, 2, cat.name
        ORDER BY 1, rental_count DESC
    """)

@app.get("/api/geography")
def get_geography():
    return run_query("""
        SELECT
            co.country,
            COUNT(DISTINCT c.customer_id) AS customer_count
        FROM customer c
        JOIN address a  ON c.address_id  = a.address_id
        JOIN city    ci ON a.city_id     = ci.city_id
        JOIN country co ON ci.country_id = co.country_id
        GROUP BY co.country
        ORDER BY customer_count DESC
        LIMIT 15
    """)

@app.get("/api/duration")
def get_duration():
    return run_query("""
        SELECT
            LEAST(ROUND(
                EXTRACT(EPOCH FROM (return_date - rental_date)) / 86400.0
            )::int, 12)                AS duration_days,
            COUNT(*)                   AS rental_count
        FROM rental
        WHERE return_date IS NOT NULL
        GROUP BY 1
        ORDER BY 1
    """)

@app.get("/api/store")
def get_store():
    return run_query("""
        SELECT
            'Store ' || s.store_id           AS store_label,
            COUNT(DISTINCT r.customer_id)    AS unique_customers,
            COUNT(DISTINCT r.rental_id)      AS rental_count,
            ROUND(SUM(p.amount)::numeric, 2) AS revenue
        FROM store     s
        JOIN inventory i ON s.store_id     = i.store_id
        JOIN rental    r ON i.inventory_id = r.inventory_id
        JOIN payment   p ON r.rental_id    = p.rental_id
        GROUP BY s.store_id
        ORDER BY s.store_id
    """)

@app.get("/api/customer/search")
def search_customer(q: str = Query(..., min_length=1)):
    return run_query("""
        SELECT
            c.customer_id,
            c.first_name || ' ' || c.last_name AS name,
            c.email,
            co.country,
            c.active
        FROM customer c
        JOIN address a  ON c.address_id  = a.address_id
        JOIN city    ci ON a.city_id     = ci.city_id
        JOIN country co ON ci.country_id = co.country_id
        WHERE LOWER(c.first_name || ' ' || c.last_name) LIKE LOWER(:pattern)
        ORDER BY name
        LIMIT 20
    """, {"pattern": f"%{q}%"})

@app.get("/api/customer/{customer_id}")
def get_customer_detail(customer_id: int):
    kpi = run_query("""
        SELECT
            COUNT(DISTINCT r.rental_id)         AS total_rentals,
            ROUND(SUM(p.amount)::numeric, 2)     AS total_spent,
            ROUND(AVG(p.amount)::numeric, 2)     AS avg_transaction,
            MIN(r.rental_date)::date             AS first_rental,
            MAX(r.rental_date)::date             AS last_rental
        FROM rental r
        JOIN payment p ON r.rental_id = p.rental_id
        WHERE r.customer_id = :cid
    """, {"cid": customer_id})

    timeline = run_query("""
        SELECT
            TO_CHAR(r.rental_date, 'Mon YYYY')  AS month,
            DATE_TRUNC('month', r.rental_date)  AS month_sort,
            COUNT(DISTINCT r.rental_id)         AS rental_count
        FROM rental r
        WHERE r.customer_id = :cid
        GROUP BY 1, 2
        ORDER BY 2
    """, {"cid": customer_id})

    genres = run_query("""
        SELECT
            cat.name                        AS genre,
            COUNT(DISTINCT r.rental_id)     AS rental_count
        FROM rental r
        JOIN inventory     i   ON r.inventory_id = i.inventory_id
        JOIN film          f   ON i.film_id       = f.film_id
        JOIN film_category fc  ON f.film_id       = fc.film_id
        JOIN category      cat ON fc.category_id  = cat.category_id
        WHERE r.customer_id = :cid
        GROUP BY cat.name
        ORDER BY rental_count DESC
        LIMIT 8
    """, {"cid": customer_id})

    history = run_query("""
        SELECT
            r.rental_id,
            r.rental_date::date     AS rental_date,
            r.return_date::date     AS return_date,
            f.title                 AS film_title,
            cat.name                AS genre,
            p.amount                AS paid
        FROM rental r
        JOIN inventory     i   ON r.inventory_id = i.inventory_id
        JOIN film          f   ON i.film_id       = f.film_id
        JOIN film_category fc  ON f.film_id       = fc.film_id
        JOIN category      cat ON fc.category_id  = cat.category_id
        JOIN payment       p   ON r.rental_id     = p.rental_id
        WHERE r.customer_id = :cid
        ORDER BY r.rental_date DESC
        LIMIT 100
    """, {"cid": customer_id})

    return {
        "kpi":      kpi[0] if kpi else {},
        "timeline": timeline,
        "genres":   genres,
        "history":  history,
    }


# ══════════════════════════════════════════════════════════════
# ML ENDPOINTS
# ══════════════════════════════════════════════════════════════

@app.get("/api/ml/revenue-forecast")
def ml_revenue_forecast():
    """
    Prediksi revenue bulan depan menggunakan Linear Regression.
    Input features: bulan ke-N, revenue bulan sebelumnya (lag), rolling avg.
    """
    try:
        rows = run_query("""
    SELECT
        TO_CHAR(DATE_TRUNC('month', r.rental_date), 'YYYY-MM') AS month,
        ROUND(SUM(p.amount)::numeric, 2)                       AS revenue,
        COUNT(DISTINCT r.rental_id)                            AS transactions
    FROM rental r
    JOIN payment p ON r.rental_id = p.rental_id
    GROUP BY 1
    ORDER BY 1
""")

        if len(rows) < 3:
            raise HTTPException(status_code=400, detail="Data terlalu sedikit untuk prediksi")

        df = pd.DataFrame(rows)
        df['revenue'] = df['revenue'].astype(float)
        df['transactions'] = df['transactions'].astype(int)

        # Feature engineering
        df['month_idx']     = range(len(df))
        df['lag_1']         = df['revenue'].shift(1)
        df['lag_2']         = df['revenue'].shift(2)
        df['rolling_avg_3'] = df['revenue'].rolling(3).mean()

        df_clean = df.dropna()

        X = df_clean[['month_idx', 'lag_1', 'lag_2', 'rolling_avg_3']].values
        y = df_clean['revenue'].values

        model = LinearRegression()
        model.fit(X, y)

        # Score
        score = model.score(X, y)

        # Prediksi bulan depan
        next_idx     = len(df)
        last_rev     = df['revenue'].iloc[-1]
        prev_rev     = df['revenue'].iloc[-2]
        rolling_avg  = df['revenue'].tail(3).mean()
        X_next       = np.array([[next_idx, last_rev, prev_rev, rolling_avg]])
        predicted    = float(model.predict(X_next)[0])

        # Prediksi 3 bulan ke depan
        predictions_3 = []
        temp_rev  = list(df['revenue'].values)
        for i in range(3):
            idx      = len(df) + i
            lag1     = temp_rev[-1]
            lag2     = temp_rev[-2]
            roll_avg = np.mean(temp_rev[-3:])
            X_f      = np.array([[idx, lag1, lag2, roll_avg]])
            pred     = float(model.predict(X_f)[0])
            predictions_3.append(round(pred, 2))
            temp_rev.append(pred)

        # Bulan terakhir sebagai referensi
        last_month_rev = float(df['revenue'].iloc[-1])
        change_pct     = round((predicted - last_month_rev) / last_month_rev * 100, 1)

        return {
            "model": "Linear Regression",
            "r_squared": round(score, 4),
            "last_month_revenue": round(last_month_rev, 2),
            "predicted_next_month": round(predicted, 2),
            "change_percent": change_pct,
            "predictions_3_months": predictions_3,
            "historical": [
                {"month": r["month"], "revenue": float(r["revenue"])}
                for r in rows
            ],
            "message": (
                f"Prediksi revenue bulan depan: ${predicted:,.2f} "
                f"({'naik' if change_pct > 0 else 'turun'} {abs(change_pct)}% dari bulan lalu)"
            )
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ML error: {str(e)}\n{traceback.format_exc()}")


@app.get("/api/ml/churn-prediction")
def ml_churn_prediction():
    """
    Prediksi customer churn menggunakan Random Forest Classifier.
    Churn = customer yang tidak rental lebih dari 90 hari dari tanggal rental terakhir di DB.
   
    Features: recency_days, frequency, monetary, avg_duration, avg_payment
    """
    try:
        rows = run_query("""
            WITH rfm AS (
                SELECT
                    c.customer_id,
                    c.first_name || ' ' || c.last_name  AS name,
                    c.email,
                    (SELECT MAX(rental_date)::date FROM rental) - MAX(r.rental_date)::date AS recency_days,
                    COUNT(DISTINCT r.rental_id)                                              AS frequency,
                    ROUND(SUM(p.amount)::numeric, 2)                                         AS monetary,
                    ROUND(AVG(p.amount)::numeric, 2)                                         AS avg_payment,
                    ROUND(AVG(
                        EXTRACT(EPOCH FROM (COALESCE(r.return_date, NOW()) - r.rental_date)) / 86400.0
                    )::numeric, 2)                                                           AS avg_duration
                FROM customer c
                JOIN rental  r ON c.customer_id = r.customer_id
                JOIN payment p ON r.rental_id   = p.rental_id
                WHERE r.rental_date IS NOT NULL
                GROUP BY c.customer_id, name, c.email
            )
            SELECT * FROM rfm
        """)

        if len(rows) < 20:
            raise HTTPException(status_code=400, detail="Data terlalu sedikit")

        df = pd.DataFrame(rows)
        for col in ['recency_days', 'frequency', 'monetary', 'avg_payment', 'avg_duration']:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # Label: churn = 1 jika recency_days > 90 (tidak rental lebih dari 90 hari)
        df['is_churn'] = (df['recency_days'] > 90).astype(int)

        if df['is_churn'].nunique() < 2:
            raise HTTPException(
                status_code=400,
                detail=f"Semua {len(df)} customer masuk satu label churn. Cek threshold atau data."
            )

        features = ['recency_days', 'frequency', 'monetary', 'avg_payment', 'avg_duration']
        X = df[features].values
        y = df['is_churn'].values

        # Train/test split
        minority = min(y.sum(), len(y) - y.sum())
        use_stratify = y if minority >= 2 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=use_stratify
        )

        # Scale
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s  = scaler.transform(X_test)

        # Random Forest
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            random_state=42,
            class_weight='balanced'
        )
        rf.fit(X_train_s, y_train)

        # Predict semua customer
        X_all_s    = scaler.transform(X)
        y_pred_all = rf.predict(X_all_s)
        proba = rf.predict_proba(X_all_s)
        y_prob_all = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
        df['churn_predicted'] = y_pred_all
        df['churn_probability'] = np.round(y_prob_all, 4)

        # Accuracy
        y_test_pred = rf.predict(X_test_s)
        accuracy    = accuracy_score(y_test, y_test_pred)

        # Feature importance
        importances = dict(zip(features, np.round(rf.feature_importances_, 4)))

        # Customers paling berisiko churn (top 15)
        at_risk = (
            df[df['churn_predicted'] == 1]
            .sort_values('churn_probability', ascending=False)
            .head(15)
        )

        # Summary stats
        total_customers = len(df)
        churn_count     = int(df['churn_predicted'].sum())
        retain_count    = total_customers - churn_count
        churn_rate      = round(churn_count / total_customers * 100, 1)

        return {
            "model": "Random Forest Classifier",
            "accuracy": round(accuracy, 4),
            "accuracy_pct": round(accuracy * 100, 1),
            "total_customers": total_customers,
            "predicted_churn": churn_count,
            "predicted_retain": retain_count,
            "churn_rate_pct": churn_rate,
            "feature_importance": importances,
            "at_risk_customers": [
                {
                    "customer_id": int(r["customer_id"]),
                    "name": r["name"],
                    "email": r["email"],
                    "recency_days": int(r["recency_days"]),
                    "frequency": int(r["frequency"]),
                    "monetary": float(r["monetary"]),
                    "churn_probability": float(r["churn_probability"])
                }
                for _, r in at_risk.iterrows()
            ],
            "message": (
                f"Dari {total_customers} customer, diprediksi {churn_count} "
                f"({churn_rate}%) akan churn. Model akurasi: {round(accuracy*100,1)}%."
            )
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ML error: {str(e)}\n{traceback.format_exc()}")


@app.get("/api/ml/summary")
def ml_summary():
    """Ringkasan status kedua model ML."""
    return {
        "models": [
            {
                "name": "Revenue Forecast",
                "algorithm": "Linear Regression",
                "endpoint": "/api/ml/revenue-forecast",
                "description": "Prediksi revenue bulan depan berdasarkan tren historis",
                "features": ["month_index", "lag_1_month", "lag_2_month", "rolling_avg_3"]
            },
            {
                "name": "Customer Churn Prediction",
                "algorithm": "Random Forest Classifier",
                "endpoint": "/api/ml/churn-prediction",
                "description": "Prediksi customer yang berpotensi tidak kembali (churn)",
                "features": ["recency_days", "frequency", "monetary", "avg_payment", "avg_duration"]
            }
        ]
    }


# ── Loyal Customer Breakdown by Region ──
@app.get("/api/rfm-by-region")
def get_rfm_by_region():
    """Breakdown loyal + champions customer count dan revenue per country."""
    return run_query("""
        WITH rfm AS (
            SELECT
                c.customer_id,
                co.country,
                (SELECT MAX(rental_date)::date FROM rental) - MAX(r.rental_date)::date AS recency_days,
                COUNT(DISTINCT r.rental_id)                                              AS frequency,
                ROUND(SUM(p.amount)::numeric, 2)                                         AS monetary
            FROM customer c
            JOIN rental  r ON c.customer_id = r.customer_id
            JOIN payment p ON r.rental_id   = p.rental_id
            JOIN address a ON c.address_id   = a.address_id
            JOIN city   ci ON a.city_id      = ci.city_id
            JOIN country co ON ci.country_id = co.country_id
            GROUP BY c.customer_id, co.country
        ),
        segmented AS (
            SELECT *,
                CASE
                    WHEN recency_days <= 30  AND frequency >= 30 AND monetary >= 150 THEN 'Champions'
                    WHEN recency_days <= 60  AND (frequency >= 20 OR monetary >= 100) THEN 'Loyal'
                    WHEN recency_days BETWEEN 61 AND 120                              THEN 'At Risk'
                    ELSE 'Lost'
                END AS segment
            FROM rfm
        )
        SELECT
            country,
            COUNT(CASE WHEN segment IN ('Loyal','Champions') THEN 1 END)  AS loyal_count,
            COUNT(CASE WHEN segment = 'Champions' THEN 1 END)             AS champions_count,
            ROUND(SUM(CASE WHEN segment IN ('Loyal','Champions') THEN monetary ELSE 0 END)::numeric, 2) AS loyal_revenue,
            COUNT(*)                                                       AS total_customers
        FROM segmented
        GROUP BY country
        HAVING COUNT(CASE WHEN segment IN ('Loyal','Champions') THEN 1 END) > 0
        ORDER BY loyal_count DESC
    """)


# ── Revenue Forecast extended (6 months) ──
@app.get("/api/ml/revenue-forecast-extended")
def ml_revenue_forecast_extended():
    """Same as revenue-forecast but predicts 6 months ahead."""
    try:
        rows = run_query("""
            SELECT
                TO_CHAR(DATE_TRUNC('month', payment_date), 'YYYY-MM') AS month,
                ROUND(SUM(amount)::numeric, 2) AS revenue,
                COUNT(*) AS transactions
            FROM payment
            GROUP BY 1
            ORDER BY 1
        """)
        if len(rows) < 4:
            raise HTTPException(status_code=400, detail="Data terlalu sedikit")

        df = pd.DataFrame(rows)
        df['revenue'] = df['revenue'].astype(float)
        df['month_idx']     = range(len(df))
        df['lag_1']         = df['revenue'].shift(1)
        df['lag_2']         = df['revenue'].shift(2)
        df['rolling_avg_3'] = df['revenue'].rolling(3).mean()
        df_clean = df.dropna()
        X = df_clean[['month_idx','lag_1','lag_2','rolling_avg_3']].values
        y = df_clean['revenue'].values
        model = LinearRegression()
        model.fit(X, y)
        score = model.score(X, y)

        # Predict 6 months ahead
        temp_rev = list(df['revenue'].values)
        predictions = []
        for i in range(6):
            idx   = len(df) + i
            lag1  = temp_rev[-1]; lag2 = temp_rev[-2]
            ravg  = float(np.mean(temp_rev[-3:]))
            pred  = float(model.predict(np.array([[idx,lag1,lag2,ravg]]))[0])
            predictions.append(round(pred, 2))
            temp_rev.append(pred)

        last = float(df['revenue'].iloc[-1])
        return {
            "model": "Linear Regression",
            "r_squared": round(score, 4),
            "last_month_revenue": round(last, 2),
            "predictions_6_months": predictions,
            "historical": [{"month": r["month"], "revenue": float(r["revenue"])} for r in rows],
            "message": f"Prediksi 6 bulan: {' → '.join(['$'+str(p) for p in predictions])}"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/customer-history/{customer_id}")
def customer_history(customer_id: int):
    return run_query("""
        SELECT 
            r.rental_date::date AS rental_date,
            r.return_date::date AS return_date,
            f.title             AS film_title,
            cat.name            AS genre,
            COALESCE(p.amount, 0) AS amount
        FROM rental r
        JOIN inventory     i   ON r.inventory_id  = i.inventory_id
        JOIN film          f   ON i.film_id        = f.film_id
        JOIN film_category fc  ON f.film_id        = fc.film_id
        JOIN category      cat ON fc.category_id   = cat.category_id
        LEFT JOIN payment  p   ON r.rental_id      = p.rental_id
        WHERE r.customer_id = :cid
        ORDER BY r.rental_date DESC
    """, {"cid": customer_id})