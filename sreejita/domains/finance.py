from sreejita.domains.base import BaseDomainDetector


class FinanceDomainDetector(BaseDomainDetector):
    domain = "finance"

    # Financial system structure (strong)
    PRIMARY_SIGNALS = {
        "account_id",
        "ledger_id",
        "transaction_type",
        "debit",
        "credit",
        "balance",
    }

    # Financial concepts (medium)
    SECONDARY_SIGNALS = {
        "expense",
        "cost",
        "liability",
        "asset",
        "equity",
        "cash_flow",
    }

    # Generic money columns (weak)
    GENERIC_SIGNALS = {
        "revenue",
        "profit",
        "amount",
        "date",
    }