import pandas as pd
from sreejita.domains.router_v2 import detect_domain
from sreejita.config import DomainEngineConfig

# Test 1: Healthcare dataset detection
df_healthcare = pd.DataFrame({
    'patient_id': [1, 2, 3],
    'admission_date': ['2024-01-01', '2024-01-02', '2024-01-03'],
    'discharge_date': ['2024-01-05', '2024-01-06', '2024-01-07'],
    'diagnosis': ['Pneumonia', 'Flu', 'Cold']
})

result = detect_domain(df_healthcare)
print(f"✅ Test 1 - Healthcare Detection: {result.domain} (confidence: {result.confidence})")
assert result.domain == "healthcare", "FAILED: Should detect as healthcare"

# Test 2: Domain hint
result_with_hint = detect_domain(df_healthcare, domain_hint="healthcare")
print(f"✅ Test 2 - With Domain Hint: {result_with_hint.domain} (confidence: {result_with_hint.confidence})")
assert result_with_hint.domain == "healthcare", "FAILED: Should respect domain hint"

# Test 3: Check config thresholds
config = DomainEngineConfig()
healthcare_threshold = config.get_min_confidence("healthcare")
print(f"✅ Test 3 - Healthcare Config Threshold: {healthcare_threshold}")
assert healthcare_threshold == 0.35, "FAILED: Should be 0.35"

print("\n✅ ALL TESTS PASSED!")
