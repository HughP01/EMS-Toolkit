if __name__ == "__main__":
    rng = np.random.default_rng(42)
 
    demo_df = pd.DataFrame(
        {
            "normal_data": rng.normal(loc=10, scale=2, size=120),
            "skewed_data": rng.exponential(scale=2, size=120),
        }
    )
 
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
 
    for ax, col in zip(axes, ["normal_data", "skewed_data"]):
        _, _, res = QQplot(demo_df, col, ax=ax)
        print(f"\n=== {col} ===")
        print(f"  Shapiro-Wilk  : W={res['shapiro_wilk']['statistic']:.4f}, "
              f"p={res['shapiro_wilk']['p_value']:.4f}  →  "
              f"{'Normal' if res['is_normal_sw'] else 'Non-normal'}")
        print(f"  Kolmogorov-Smirnov: D={res['kolmogorov_smirnov']['statistic']:.4f}, "
              f"p={res['kolmogorov_smirnov']['p_value']:.4f}  →  "
              f"{'Normal' if res['is_normal_ks'] else 'Non-normal'}")
 

    plt.show()
