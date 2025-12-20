# sreejita/domains/intelligence/domain_intents.py

DOMAIN_INTENTS = {

    # =====================================================
    # SUPPLY CHAIN 🚚 (Logistics, Inventory, Procurement)
    # =====================================================
    "supply_chain": {
        "high": {
            # Logistics
            "warehouse", "carrier", "shipping_mode", "freight",
            "delivery_status", "tracking_number", "route",
            
            # Inventory
            "inventory_level", "stock_on_hand", "restock", 
            "safety_stock", "reorder_point",
            
            # Suppliers
            "supplier", "vendor", "procurement",
            
            # Performance
            "lead_time", "cycle_time", "on_time_delivery",
            "fill_rate", "backorder"
        },
        "ambiguous": {
            "ship_date", "order_id", "quantity", "location", 
            "return", "sku", "item", "weight"
        }
    },

    # =====================================================
    # MANUFACTURING 🏭 (Machines, IoT, Production)
    # =====================================================
    "manufacturing": {
        "high": {
            "machine_id", "production_line", "batch_number",
            "defect_rate", "yield", "uptime", "downtime",
            "maintenance", "sensor", "temperature", "pressure",
            "vibration", "factory", "plant", "shift_id",
            "scrap", "rework"
        },
        "ambiguous": {
            "date", "status", "operator", "output", "count", "quality"
        }
    },

    # =====================================================
    # HR / WORKFORCE 👥
    # =====================================================
    "hr": {
        "high": {
            # Identity
            "employee", "employee_id", "staff", "personnel",

            # Org structure
            "department", "designation", "role", "manager",

            # Compensation
            "salary", "compensation", "ctc", "payroll", "bonus",

            # Lifecycle
            "attrition", "termination", "resignation", "hire_date",
            "joining_date", "exit_date",

            # Performance & attendance
            "performance_score", "rating", "leave_balance", "timesheet"
        },
        "ambiguous": {
            "id", "date", "status", "gender", "age", "location", 
            "name", "email"
        }
    },

    # =====================================================
    # MARKETING 📢
    # =====================================================
    "marketing": {
        "high": {
            "campaign", "ad_group", "creative", "audience",
            "impressions", "clicks", "ctr", "cpc", "cpm",
            "roas", "ad_spend", "cost_per_acquisition",
            "campaign_id", "utm_source"
        },
        "ambiguous": {
            "channel", "source", "conversion", "cost", 
            "revenue", "keyword", "budget"
        }
    },

    # =====================================================
    # RETAIL 🛍️
    # =====================================================
    "retail": {
        "high": {
            "store_id", "store_location", "cashier", "pos_id",
            "shelf", "aisle", "zone", "markdown", "promotion",
            "loyalty_card", "foot_traffic", "basket_size"
        },
        "ambiguous": {
            "store", "sales", "category", "item", 
            "quantity", "profit", "price", "sku"
        }
    },

    # =====================================================
    # ECOMMERCE 🛒
    # =====================================================
    "ecommerce": {
        "high": {
            "cart_abandonment", "add_to_cart", "checkout",
            "conversion_rate", "aov", "cac", "session_duration",
            "bounce_rate", "pageviews", "unique_visitors",
            "payment_gateway", "shipping_method"
        },
        "ambiguous": {
            "session_id", "referrer", "ip_address", "discount_code",
            "user_id", "order_date", "product_view"
        }
    },

    # =====================================================
    # CUSTOMER (CRM) 🤝
    # =====================================================
    "customer": {
        "high": {
            "customer_id", "customer_name", "segment", 
            "recency", "frequency", "monetary", "rfm",
            "lifetime_value", "churn", "nps", "csat",
            "support_ticket"
        },
        "ambiguous": {
            "purchase_date", "location", "device", "channel",
            "transaction_id", "amount", "email", "phone"
        }
    },

    # =====================================================
    # FINANCE 💰 (Corporate + Market)
    # =====================================================
    "finance": {
        "high": {
            # Corporate Finance
            "portfolio", "asset", "liability", "equity",
            "dividend", "interest_rate", "ticker",
            "transaction_type", "balance", "net_income", 
            "cash_flow", "ebitda", "volatility", "beta",

            # Market Data (OHLCV)
            "open", "close", "high", "low",
            "adj_close", "adjusted_close",
            "volume", "market_cap",
            "returns"
        },
        "ambiguous": {
            "date", "value", "amount", "price", "currency"
        }
    },

    # =====================================================
    # SALES (B2B) 💼
    # =====================================================
    "sales": {
        "high": {
            "opportunity", "pipeline", "deal_stage", "quota",
            "account_executive", "lead_source", "prospect",
            "win_rate", "sales_rep", "closing_date",
            "contract_value", "arr", "mrr", "bookings"
        },
        "ambiguous": {
            "amount", "probability", "status", "owner", 
            "account", "territory", "contact"
        }
    },

    # =====================================================
    # HEALTHCARE 🏥
    # =====================================================
    "healthcare": {
        "high": {
            "diagnosis", "patient", "treatment", "admission",
            "discharge", "doctor", "hospital", "clinical",
            "readmission", "mortality", "insurance_provider", 
            "blood_type", "systolic", "diastolic", "bmi"
        },
        "ambiguous": {
            "id", "date", "gender", "age", "status", 
            "weight", "height", "visit"
        }
    },

    # =====================================================
    # EDUCATION 🎓
    # =====================================================
    "education": {
        "high": {
            "student_id", "gpa", "course_id", "semester",
            "grade", "faculty", "enrollment", "major",
            "exam_score", "attendance_rate", "graduation_year"
        },
        "ambiguous": {
            "name", "age", "gender", "subject", "class", "score"
        }
    }
}

