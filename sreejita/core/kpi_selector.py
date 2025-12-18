def select_ranked_kpis(df, registry, max_kpis=4):
    selected = []

    for item in sorted(registry, key=lambda x: x["rank"]):
        if all(col in df.columns for col in item["required_columns"]):
            selected.append(item)
        if len(selected) == max_kpis:
            break

    return selected
