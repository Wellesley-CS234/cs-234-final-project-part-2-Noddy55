import streamlit as st
import pandas as pd
import numpy as np


st.set_page_config(page_title="US vs UK: Current Events on Wikipedia",layout="wide")


PATH = "us_uk_week_with_ml_event_label.csv"
PATH2="us_uk_week_labeled_events.csv"
df = pd.read_csv(PATH)
df2= pd.read_csv(PATH2)
# i will convert the column to date time because i need to use it later
df["date"] = pd.to_datetime(df["date"])




#ALL TABS
intro, data_sum, features, classification, hypo, viz = st.tabs(["✨ Introduction", #this is to create tabs 
        "📝 Data Summary",
        "⚙️ New Features",
        "🔠 Text Classification",
        "🔎 Hypothesis Testing",
        "📊 Interactive Visualization"])

#INTRODUCTION PART
with intro:
   
    st.markdown( """<h1 style="color:#a4dbdb; font-size: 48px; ">
            Do Wikipedia Readers in the US and UK Differ in Their Interest in Current Events?</h1>""", unsafe_allow_html=True,) # had to add this as they treat html as text
    left, right = st.columns([2, 1]) # these are the two columns named

    with left:
         st.write("""
### Introduction: 
Being online in the current day and age exposes us to a variety of opinions, world views, and stereotypes. One such persistent stereotype circulating in online discourse is that Americans are less engaged with current affairs and global news than Europeans or the rest of the world.  
This is what inspired me to start this project. Wikipedia acts as a proxy for interest, being one of the most commonly visited sites for information for all kinda of general information.  
However, rather than assuming the stereotype is correct, my goal is to test whether there is actually a difference instead.

                  
### Expectation:
Based on online discourse, I expect that U.S. readers may show slightly lower engagement, but the data may reveal otherwise.

This project uses one week of Wikipedia pageviews (`Feb 6–12, 2023`), focusing on the English Wikipedia(en) for both countries to keep the comparison consistent         
                  """)
        

    with right:
        st.metric("Countries",f"GB | US")
        st.metric("Number of rows (articles × days)", len(df))
        st.metric("Date range", "2023-02-06 - 2023-02-12")

#DATA SUM
with data_sum:
    st.write( """<h1 style="color:#eb83a2; font-size: 48px; text-align: center;">Data Summary 🔍</h1>""",unsafe_allow_html=True,)
    left2, right2 = st.columns([2, 1])
    with left2:
         st.write( """<h1 style="color:#81b3a9; font-size: 30px;">Stats and Limitations</h1>""",unsafe_allow_html=True,)
         st.write("""
In order to measure **interest in current events**, I use the **proportion of Wikipedia pageviews** that go to articles classified as *current events*.
If a higher share of pages viewed are about current events, then I will interpret that as higher proportion of people in that county being interested in current events.

To keep the sample small, I have decided to focus on a single, relatively uneventful week in both the U.S. and U.K.:
- **Date range:** `2023-02-06` to `2023-02-12`  
- **Countries:** United States [`US`] and Great Britain [`GB`]  
- **wikipedia:** English Wikipedia (`en`)""")
         
         with st.expander("⚠️ Key limitations", expanded=False):
             st.write(
            """
            -  This dataset only works with engagement on the english wikipedia(`en.wikipedia') articles. This is due to english being the most-commonly spoken language in both regions. Canada could have been chosen as well, but the large number of french speakers made me choose the UK(GB).
               This is reasonable because English is widely used in both US and GB,  but it ignores readers using other language editions.
            - The **current-events label is subjective**. I generated a keyword list with using genAI and applied it to ~4000 unique "instance_of" labels to build a **ground-truth event vs non-event**.
            - Some articles (e.g. `**people**`, `**artifacts**`, `**Wikimedia main pages**`) and missing labels were **filtered out**.
            - My week is just a **snapshot** of one in many weeks. A different week (e.g., around elections or major global/internal crises) could produce different patterns.""")
    with right2: 
        st.write("""<h3 style="color:#F29E4C; font-size: 30px;">Major steps </h3>""",unsafe_allow_html=True,)
        st.write("""

Here are some of the major steps that were involved in this project:
- Using provided file to find the `instance_of` labels.
- Use those labels to define a "ground-truth" for the current-event category.
- Training a Naive Bayes and Zero Shot text classifier to predict whether an article is a current event or not.
- Compare US vs GB using hypothesis testing on the resulting 0/1 label.""")


        st.write("""This dataset comes from a DPDP database for Wikipedia pageviews.  
After filtering, each row has the following:

- **`date`**
- **`country`** (**`US`** or **`GB`**)
- **`project`** (`en.wikipedia`)
- **`article title`**()
- **`QID`**
- **`pageviews`** on that date (`views`)

I then merged this with the attributes to get more information about each article like labels.""")

    left, right = st.columns(2)

    with left:
        st.write("""<h3 style="color:#c491d9; font-size: 30px;">Basic structure of the df </h3>""",unsafe_allow_html=True,)

        st.write(df.head())

    with right:
        st.write("""<h3 style="color:#b8cf82; font-size: 30px;">Descriptive statistics of pageviews </h3>""",unsafe_allow_html=True,)
        st.write(df["views"].describe())
   
    showcase=df.groupby("country_code").agg(number_of_rows=("article", "size"),number_of_unique_articles=("article", "nunique")).reset_index()



# NEW FEATURES OR COLUMNS I ADDED
with features:
    st.write("""<h1 style="color:#e0c58b; font-size: 48px;text-align: center;">New Features Added to the Dataset 🛠️</h1>""",unsafe_allow_html=True)
    st.write("NOTE: All data in tables has been arranged in descending order of pageviews.")
    left, middle, right = st.columns(3)

    
    with left:
      st.write("""
        <h1 style="color:#fa8f87; font-size: 30px;text-align: center;">is_event_base_truth </h1>""",unsafe_allow_html=True)
    
      st.write("""
Using a list of event-related and lowercased strings generated after multiple iteratins of feeding all unique lables to genAI,  
I marked an article as a current event if its "instance_of" contained any of the generated keywords((Ex:-. "war", "election", "hurricane", "Super Bowl", "award ceremony", "bombing", etc.)

- is_event_base_truth --> `1` --> some kind of current event
- is_event_base_truth --> `0` --> everything else
    
This used to establish a “ground truth” to train classifier in the upcoming steps.""")
      st.write(df2[[ "country_code", "article", "is_event_base_truth"]].head())

    with middle:
       st.write("""
        <h1 style="color:#87faf8; font-size: 30px;text-align: center;">text</h1>""",unsafe_allow_html=True)
        
       st.write("""       
To prepare text for the classifier, I combined the article name and description and lowercased and also removed punctuation to create a singular "text" field.""")
       st.write(df[[ "article", "description", "text"]].head())

    with right:
        st.write("""
        <h1 style="color:#e67cde; font-size: 30px; text-align: center;">is_event_ml_pred</h1>""",unsafe_allow_html=True)
        st.write("""
Finally, I trained a Multinomial Naive Bayes model on the text to predict whether an article is an event (`1`) or not (`0`) for the entire huge dataset, using "is_event_base_truth" as actual labels for comparison.
This is the column used later for hypothesis testing and visualizations.""")
        st.write(df[["country_code", "article","is_event_ml_pred"]].head())

    st.subheader("Example of dataset after these changes")
    st.write("### Top 5")
    st.write(df[["date", "country_code", "article", "views", "instance_of", "is_event_ml_pred"]].head())
    st.write("### Bottom 5")
    st.write(df[["date", "country_code", "article", "views", "instance_of", "is_event_ml_pred"]].tail())
    

#TEXT CLASS
with classification:
    st.write("""<h1 style="color:#7fb3ff; font-size: 48px; text-align: center;">Text Classification 🔠</h1>""", unsafe_allow_html=True)
    left, right = st.columns([1.2, 1])
    with left:
        st.write("### What I classified")
        st.write("- **Task**: Current event (`1`) vs Other Non- curnent (`0`)")
        st.write("- **Input text**: `article + description` → cleaned & lowercased(pre-processing)")
        st.write("- **Model**: Multinomial Naive Bayes (using the class notebook tutorial)")

        with st.expander("Show the steps↓↓↓↓"):
            st.write("""
- Build text = article + description
- Regex and cleanup (remove punctuation)
- Tokenize + stemming
- CountVectorizer 
- Train/test split (80/20)
- MultinomialNB fit 
- Evaluate -- all of this from the noetbook
            """)
    with right:
        st.write("### Results")
        st.metric("Accuracy", "0.977096")# this is from my notebook that has all of the data and files
        st.metric("F1 score", "0.977786")
    st.write(df)


#HYPOTHESIS test

with hypo:
    st.write("""<h1 style="color:#50C878; font-size: 48px; text-align: center;">Hypothesis Testing 🧪</h1>""", unsafe_allow_html=True)

    st.write("""### RQ

"Do Wikipedia readers in the US and UK differ in their interest in current events?
             
#### Measurement :
We use the proportion of articles classified as current events in the column (is_event_ml_pred = 1) among all pages viewed in each country during that week.

#### Null hypothesis (`H0`):
The US and UK have equal mean proportions of event-page views.

#### Alternative hypothesis (`H1`):
The US and UK have different mean proportions of event-page views.
             
#### Even though the US and UK show slightly different proportions of event-related articles (about 8.5% in GB versus 8.1% in the US), the absolute difference is less than half a percentage point. The two-sample t-test returns a very small p-value (p <0.0005754450891545388). Because the dataset is extremely large , even small numerical differences become very statistically significant.

### Process

- I took all rows from one week (`2023-02-06` to `2023-02-12`)  
- The mean of this 0/1 column (is_event_ml_pred) is the **proportion** of event pages for that country.  
- I used a **two-sample t-test** on the two sets of 0/1 values to compare the mean between US and GB.
             

### Proportions
- **GB event share:** `8.596`%  
- **US event share:** `8.196`%  
- **Difference:** ~0.40 percentage points
             
### Statistical Test results
             
- **t-statistic:** -3.443  
- **p-value:** `0.000575`  

We reject the null hypothesis of equal proportions. 

However, since the less than 0.5 percentage point difference is extremely small, there seems to be no meaningful difference in interest between US and UK readers.

`Conclusion: The stereotype that US readers are less interested in current events is not supported by the data.`""")
    
    
# used help from ai on this part to knoe all of the differnt functions etc.

import altair as alt
#VIZ
with viz:
    st.header("Top Articles: US vs. GB")
    top = st.slider("Number of top articles per country:",min_value=10,max_value=100,step=1,value=10)
#shows the top 10-100 articles for each country and their composition as current or non currnet event
    aggr = df.groupby(["country_code", "article"])["views"].sum().reset_index(name="total_views") #aggregate or groupby by two params since diff countries and need pageviews
    aggr["is_event"] = df.groupby(["country_code", "article"])["is_event_ml_pred"].max().values
    aggr["event_label"] = aggr["is_event"].replace({1: "Current Event", 0: "Other"}) # giving the 0 and 1s names to make more sense of it when displayed

#usaa
    us_top = aggr[aggr["country_code"] == "US"].sort_values("total_views", ascending=False).head(top)# uses the top as slider input here
     
    us_chart = (alt.Chart(us_top).mark_bar().encode(
        x=alt.X("total_views:Q", title="Total views over the week"),
        color=alt.Color("event_label:N",scale=alt.Scale(range=["#F7C5CC", "#8FC7E6"])),
        y=alt.Y("article:N", sort="-x", title="Article"),
        tooltip=["article", "total_views", "event_label"]))# for the hovering thing
    st.subheader(f"Top {top} Wikipedia articles in the US by total views")


    st.altair_chart(us_chart)

 #GB
 # doing the same for GB
    gb_top = aggr[aggr["country_code"] == "GB"].sort_values("total_views", ascending=False).head(top)

    gb_chart = (alt.Chart(gb_top).mark_bar().encode(x=alt.X("total_views:Q", title="Total views over the week"),
        y=alt.Y("article:N", sort="-x", title="Article"),
        color=alt.Color("event_label:N",scale=alt.Scale(range=["#C7E9C0", "#D7C9FF"])),
        tooltip=["article", "total_views", "event_label"]))# for the hovering thinig
    st.subheader(f"Top {top} Wikipedia articles in GB by total views")
    st.altair_chart(gb_chart, use_container_width=True)


# second viz
    st.subheader("How Much Attention Goes to Current Events?")

    prop = (df.groupby(["country_code", "is_event_ml_pred"], as_index=False)["views"].sum())
    prop["share"] = prop["views"] / prop.groupby("country_code")["views"].transform("sum")
    prop["type"] = prop["is_event_ml_pred"].replace({1: "Current Event", 0: "Other"}) # labeling to make sense of 0 and 1
    chart1 = (alt.Chart(prop).mark_bar().encode(
        x=alt.X("country_code:N", title="Country"),
        y=alt.Y("share:Q", title="Share of total views"),
        color=alt.Color("type:N", title="Type"),
        tooltip=["country_code", "type", alt.Tooltip("share:Q", format=".2%")]))
        
    st.altair_chart(chart1, use_container_width=True)



    
# viz 3

    st.subheader("Daily attention for a chosen event type")

    event_labels = (df[df["is_event_ml_pred"] == 1]["instance_of"].value_counts().index.tolist())# creating. alist with all of the instace_of labels
    
    chosen = st.selectbox("Choose an event type (from `instance_of` labels):",event_labels) #will prompt viewer to select
    
    d = df[(df["is_event_ml_pred"] == 1) &(df["instance_of"] == chosen)] # getting the data onluy for the chosen topic
    
    
    #total views every day across every category for both countries
    daily_total = (df.groupby(["date", "country_code"], as_index=False)["views"].sum().rename(columns={"views": "total_views"}))
    #total views for  chosen  per day both country
     
    daily_event = (d.groupby(["date", "country_code"], as_index=False)["views"].sum().rename(columns={"views": "event_views"}))
    
    # merging
    daily = daily_event.merge(daily_total,on=["date", "country_code"])

    daily["share"] = daily["event_views"] / daily["total_views"]# dividing to find proportion

    line = (alt.Chart(daily).mark_line(point=True).encode(x=alt.X("date:T", title="Day"),
        y=alt.Y("share:Q", title="Share of daily Wikipedia views"),
        color=alt.Color("country_code:N",scale=alt.Scale(range=["#FF6B6B", "#4D96FF"])),
        tooltip=["date:T","country_code"]))
    
    st.subheader(f"Daily share of attention for {chosen}")

    st.altair_chart(line, use_container_width=True)
