from sreejita.domains.base import BaseDomainDetector


class FinanceDomainDetector(BaseDomainDetector):
    domain = "finance"

    PRIMARY_SIGNALS = {
        "account_id",
        "ledger_id",
        "transaction_type",
        "debit",
        "credit",
        "balance",
    }

    SECONDARY_SIGNALS = {
        "expense",
        "cost",
        "liability",
        "asset",
        "equity",
        "cash_flow",
    }

    GENERIC_SIGNALS = {
        "revenue",
        "profit",
        "amount",
        "date",
    }
