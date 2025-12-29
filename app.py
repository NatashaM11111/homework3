import streamlit as st
import pandas as pd
import altair as alt
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

# ---------- 1. Load data ----------

@st.cache_data
def load_data():
    products = pd.read_csv("products.csv")
    testimonials = pd.read_csv("testimonials.csv")
    reviews = pd.read_csv("reviews_with_sentiment.csv")

    # Parse review dates
    if "date_parsed" in reviews.columns:
        reviews["date_parsed"] = pd.to_datetime(reviews["date_parsed"], errors="coerce")
    elif "date_raw" in reviews.columns:
        reviews["date_parsed"] = pd.to_datetime(reviews["date_raw"], errors="coerce")
    else:
        for col in reviews.columns:
            if "date" in col.lower():
                reviews["date_parsed"] = pd.to_datetime(reviews[col], errors="coerce")
                break

    return products, testimonials, reviews


products_df, testimonials_df, reviews_df = load_data()

# ---------- 2. Sidebar navigation ----------

st.sidebar.title("Navigation")
section = st.sidebar.radio(
    "Select section:",
    ("Products", "Testimonials", "Reviews")
)

st.title("Web Scraping Dev – Data Explorer")

# ---------- 3. Products section ----------

if section == "Products":
    st.header("Products")
    st.dataframe(products_df)

# ---------- 4. Testimonials section ----------

elif section == "Testimonials":
    st.header("Testimonials")
    st.dataframe(testimonials_df)

# ---------- 5. Reviews section (core feature) ----------

elif section == "Reviews":
    st.header("Reviews")

    if "date_parsed" not in reviews_df.columns:
        st.error("No parsed date column found in reviews data.")
    else:
        # Keep valid dates only
        reviews_valid = reviews_df.dropna(subset=["date_parsed"]).copy()
        reviews_2023 = reviews_valid[
            reviews_valid["date_parsed"].dt.year == 2023
        ].copy()

        if reviews_2023.empty:
            st.warning("No reviews found for 2023.")
        else:
            reviews_2023["year_month"] = reviews_2023["date_parsed"].dt.to_period("M")

            # January–May 2023 (based on available data)
            month_options = pd.period_range(start="2023-01", end="2023-05", freq="M")
            month_display = [m.strftime("%B %Y") for m in month_options]
            month_map = dict(zip(month_display, month_options))

            selected_month_label = st.select_slider(
                "Select a month in 2023:",
                options=month_display,
                value=month_display[0]
            )

            selected_period = month_map[selected_month_label]
            filtered = reviews_2023[
                reviews_2023["year_month"] == selected_period
            ].copy()

            st.subheader(f"Reviews for {selected_month_label}")
            st.write(f"Number of reviews: {len(filtered)}")

            if filtered.empty:
                st.info("No reviews for this month.")
            else:
                # Sentiment already computed locally using Hugging Face Transformers
                if "sentiment" in filtered.columns:
                    st.subheader("Sentiment summary")

                    # Normalize sentiment labels to lowercase strings
                    sentiments = filtered["sentiment"].astype(str).str.lower()

                    # Ensure both categories are always present (positive, negative)
                    counts = (
                        sentiments.value_counts()
                        .reindex(["positive", "negative"], fill_value=0)
                        .reset_index()
                    )
                    counts.columns = ["sentiment", "count"]

                    # Order categories for display
                    counts["sentiment"] = pd.Categorical(
                        counts["sentiment"], categories=["positive", "negative"], ordered=True
                    )

                    # Compute average confidence (hf_score) per sentiment if available
                    if "hf_score" in filtered.columns:
                        scores = filtered.copy()
                        scores["sentiment"] = sentiments
                        avg = (
                            scores.groupby("sentiment")["hf_score"]
                            .mean()
                            .reindex(["positive", "negative"], fill_value=0)
                            .reset_index()
                        )
                        avg.columns = ["sentiment", "avg_score"]
                    else:
                        avg = pd.DataFrame({"sentiment": ["positive", "negative"], "avg_score": [0.0, 0.0]})

                    # Merge average score into counts for tooltip
                    counts = counts.merge(avg, on="sentiment")

                    # Show counts as simple metrics
                    pos_count = int(counts.loc[counts["sentiment"] == "positive", "count"].iat[0])
                    neg_count = int(counts.loc[counts["sentiment"] == "negative", "count"].iat[0])
                    c1, c2 = st.columns(2)
                    c1.metric("Positive", pos_count)
                    c2.metric("Negative", neg_count)

                    # Show average confidence as percentage metrics
                    pos_avg = float(counts.loc[counts["sentiment"] == "positive", "avg_score"].iat[0])
                    neg_avg = float(counts.loc[counts["sentiment"] == "negative", "avg_score"].iat[0])
                    d1, d2 = st.columns(2)
                    d1.metric("Avg Confidence (Positive)", f"{pos_avg:.1%}")
                    d2.metric("Avg Confidence (Negative)", f"{neg_avg:.1%}")

                    # Color scale: negative = red, positive = green
                    color_scale = alt.Scale(domain=["negative", "positive"], range=["#d62728", "#2ca02c"])

                    # Chart with tooltip showing average confidence
                    chart = (
                        alt.Chart(counts)
                        .mark_bar()
                        .encode(
                            x=alt.X("sentiment:N", sort=["positive", "negative"], title="Sentiment"),
                            y=alt.Y("count:Q", title="Count"),
                            color=alt.Color("sentiment:N", scale=color_scale, legend=None),
                            tooltip=[
                                alt.Tooltip("sentiment:N", title="Sentiment"),
                                alt.Tooltip("count:Q", title="Count"),
                                alt.Tooltip("avg_score:Q", format=".1%", title="Avg Confidence"),
                            ],
                        )
                        .properties(width=480, height=300)
                    )

                    st.altair_chart(chart, use_container_width=True)
                else:
                    st.warning(
                        "Sentiment columns not found. "
                        "Make sure you are loading reviews_with_sentiment.csv."
                    )

                # Display review table
                columns_to_show = [
                    c for c in ["date_parsed", "text", "stars", "sentiment", "hf_score"]
                    if c in filtered.columns
                ]

                st.dataframe(filtered[columns_to_show])

                # Word Cloud (bonus)
                st.subheader("Word Cloud (Bonus)")

                text_blob = " ".join(filtered["text"].dropna().astype(str).tolist())

                if text_blob.strip():
                    wc = WordCloud(
                        width=800,
                        height=400,
                        background_color="white",
                        stopwords=STOPWORDS
                    ).generate(text_blob)

                    fig, ax = plt.subplots(figsize=(10, 5))
                    # Use the PIL image to avoid WordCloud.__array__ calling np.asarray(copy=...)
                    ax.imshow(wc.to_image(), interpolation="bilinear")
                    ax.axis("off")
                    st.pyplot(fig)
                else:
                    st.info("Not enough review text to generate a word cloud for this month.")
