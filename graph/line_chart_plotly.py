import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Load data
df = pd.read_csv("csv/movie_dataset.csv")

# Prepare data
df["release_date"] = pd.to_datetime(df["release_date"])
df["release_year"] = (df["release_date"].dt.year).astype("int64", errors="ignore")

# Group by year and sum numeric columns
df_years = df.groupby(by=["release_year"]).sum(numeric_only=True)
df_years.sort_values(by=["release_year"], ascending=True, inplace=True)
df_years = df_years.iloc[:-2]  # Remove incomplete rows

# Reset index to make release_year a column
df_years = df_years.reset_index()

# ========== Chart 1: Simple Revenue Line Chart ==========
fig1 = px.line(
    df_years,
    x="release_year",
    y="revenue",
    title="Revenue Over Time (Interactive)",
    labels={"release_year": "Year", "revenue": "Revenue ($)"},
    markers=True,
)

fig1.update_layout(
    hovermode="x unified", height=500, xaxis_title="Release Year", yaxis_title="Revenue ($)", template="plotly_white"
)

fig1.update_traces(line=dict(color="#1f77b4", width=3), marker=dict(size=8))

# Save to HTML
fig1.write_html("revenue_over_time.html")
print("Saved: revenue_over_time.html")


# ========== Chart 2: Multi-line Chart (Revenue & Budget) ==========
fig2 = go.Figure()

# Add Revenue line
fig2.add_trace(
    go.Scatter(
        x=df_years["release_year"],
        y=df_years["revenue"],
        mode="lines+markers",
        name="Revenue",
        line=dict(color="#1f77b4", width=3),
        marker=dict(size=8),
        hovertemplate="Year: %{x}<br>Revenue: $%{y:,.0f}<extra></extra>",
    )
)

# Add Budget line
fig2.add_trace(
    go.Scatter(
        x=df_years["release_year"],
        y=df_years["budget"],
        mode="lines+markers",
        name="Budget",
        line=dict(color="#ff7f0e", width=3),
        marker=dict(size=8),
        hovertemplate="Year: %{x}<br>Budget: $%{y:,.0f}<extra></extra>",
    )
)

fig2.update_layout(
    title="Revenue & Budget Over Time",
    xaxis_title="Release Year",
    yaxis_title="Value ($)",
    hovermode="x unified",
    height=600,
    template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

# Save to HTML
fig2.write_html("revenue_budget_over_time.html")
print("Saved: revenue_budget_over_time.html")


# ========== Chart 3: Area Chart - Cumulative Revenue ==========
df_years["cumulative_revenue"] = df_years["revenue"].cumsum()

fig3 = go.Figure()

fig3.add_trace(
    go.Scatter(
        x=df_years["release_year"],
        y=df_years["cumulative_revenue"],
        mode="lines",
        name="Cumulative Revenue",
        fill="tozeroy",
        line=dict(color="#9467bd", width=2),
        hovertemplate="Year: %{x}<br>Cumulative Revenue: $%{y:,.0f}<extra></extra>",
    )
)

fig3.update_layout(
    title="Cumulative Revenue Over Time",
    xaxis_title="Release Year",
    yaxis_title="Cumulative Revenue ($)",
    hovermode="x",
    height=500,
    template="plotly_white",
)

# Save to HTML
fig3.write_html("cumulative_revenue_over_time.html")
print("Saved: cumulative_revenue_over_time.html")

print("\n All charts saved successfully!")
print("Files created:")
print("  - revenue_over_time.html")
print("  - revenue_budget_over_time.html")
print("  - cumulative_revenue_over_time.html")
