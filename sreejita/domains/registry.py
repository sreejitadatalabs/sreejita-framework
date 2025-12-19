class DomainRegistry:
    def __init__(self):
        self._domains = {}

    def register(self, name, domain_cls):
        self._domains[name.lower()] = domain_cls

    def get_domain(self, name):
        cls = self._domains.get(name.lower())
        if not cls:
            return None
        return cls()

    def list_domains(self):
        return list(self._domains.keys())


# SINGLETON
registry = DomainRegistry()
