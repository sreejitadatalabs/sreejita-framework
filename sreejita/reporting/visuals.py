import matplotlib.pyplot as plt

def shipping_cost_vs_sales(df, output_path):
    plt.figure()
    plt.scatter(df["sales"], df["shipping_cost"])
    plt.xlabel("Sales")
    plt.ylabel("Shipping Cost")
    plt.title("Shipping Cost vs Sales")
    plt.savefig(output_path)
    plt.close()
