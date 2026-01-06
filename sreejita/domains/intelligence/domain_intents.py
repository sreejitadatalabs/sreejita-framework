# =====================================================
# DOMAIN INTENTS — CANONICAL & NORMALIZATION-ALIGNED
# Sreejita Framework v3.6 (STABLE)
# =====================================================

DOMAIN_INTENTS = {

    # =====================================================
    # SUPPLY CHAIN 🚚
    # =====================================================
    "supply_chain": {
        "high": {
            "warehouse", "carrier", "shipping_mode", "freight",
            "delivery_status", "tracking_number", "route",
            "inventory_level", "stock_on_hand", "safety_stock",
            "reorder_point", "supplier", "vendor", "procurement",
            "lead_time", "cycle_time", "on_time_delivery",
            "fill_rate", "backorder", "sku"
        },
        "ambiguous": {
            "order_id", "quantity", "location", "date"
        }
    },

    # =====================================================
    # HR 👥
    # =====================================================
    "hr": {
        "high": {
            "employee_id", "employee_name", "staff_id",
            "department", "designation", "role", "manager",
            "salary", "compensation", "ctc", "payroll", "bonus",
            "attrition", "termination", "resignation",
            "hire_date", "joining_date", "exit_date",
            "performance_score", "rating",
            "leave_balance", "attendance", "timesheet"
        },
        "ambiguous": {
            "id", "date", "gender", "age", "location"
        }
    },

    # =====================================================
    # MARKETING 📢
    # =====================================================
    "marketing": {
        "high": {
            "campaign_id", "ad_group", "creative_id",
            "impressions", "clicks", "ctr", "cpc", "cpm",
            "roas", "ad_spend", "cost_per_acquisition",
            "utm_source", "utm_medium"
        },
        "ambiguous": {
            "channel", "cost", "revenue", "date"
        }
    },

    # =====================================================
    # RETAIL 🛍️
    # =====================================================
    "retail": {
        "high": {
            "store_id", "store_location", "pos_id",
            "cashier", "shelf", "aisle", "zone",
            "promotion", "markdown", "loyalty_card",
            "foot_traffic", "basket_size"
        },
        "ambiguous": {
            "sales", "quantity", "price", "sku", "date"
        }
    },

    # =====================================================
    # ECOMMERCE 🛒
    # =====================================================
    "ecommerce": {
        "high": {
            "cart_abandonment", "add_to_cart",
            "checkout", "conversion_rate",
            "aov", "cac", "session_duration",
            "bounce_rate", "pageviews",
            "unique_visitors", "payment_gateway",
            "shipping_method"
        },
        "ambiguous": {
            "user_id", "order_date", "discount_code", "amount"
        }
    },

    # =====================================================
    # CUSTOMER 🤝
    # =====================================================
    "customer": {
        "high": {
            "customer_id", "customer_name", "segment",
            "rfm", "recency", "frequency", "monetary",
            "lifetime_value", "churn", "nps", "csat",
            "support_ticket"
        },
        "ambiguous": {
            "email", "phone", "transaction_id", "amount", "date"
        }
    },

    # =====================================================
    # FINANCE 💰
    # =====================================================
    "finance": {
        "high": {
            "revenue", "expense", "profit", "loss",
            "asset", "liability", "equity",
            "cash_flow", "net_income", "ebitda",
            "interest_rate", "balance",
            "open", "close", "high", "low",
            "adjusted_close", "volume", "market_cap"
        },
        "ambiguous": {
            "price", "amount", "currency", "date"
        }
    },

    # =====================================================
    # HEALTHCARE 🏥 (CRITICAL FIX)
    # =====================================================
    "healthcare": {
        "high": {
            # Identity & encounter
            "patient_id", "encounter", "visit_id",

            # Dates
            "admission_date", "discharge_date", "fill_date",

            # Clinical
            "diagnosis", "treatment", "doctor", "bed_id",

            # Operations
            "length_of_stay", "los", "duration",

            # Outcomes
            "readmitted", "mortality", "flag",

            # Pharmacy
            "days_supply", "supply", "rx_volume",

            # Financial healthcare
            "billing_amount", "cost",

            # Population health
            "population"
        },
        "ambiguous": {
            "id", "date", "age", "gender", "facility", "status"
        }
    },
}
