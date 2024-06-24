import pandas as pd
import datetime as dt
pd.set_option('display.max_columns', None)
pd.set_option('display.width',500)
# pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df_ = pd.read_csv("/kaggle/input/flo-data-20k/flo_data_20k.csv")
df = df_.copy()

def check_df(dataframe, head = 5):
    print("########## SHAPE ##########")
    print(dataframe.shape)
    print("########## TYPES ##########")
    print(dataframe.dtypes)
    #print("########## HEAD ##########")
    #print(dataframe.head(head))
    #print("########## TAIL ##########")
    #print(dataframe.tail(head))
    print("########## NA ##########")
    print(dataframe.isnull().sum())
    #print("########## QUANTILES ##########")
    #print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1])),

check_df(df)

def variable_analysis(dataframe):
    print("########## HOW MANY UNIQUE USER ##########")
    print(dataframe["master_id"].nunique())
    print("########## HOW MANY UNIQUE CHANNEL ##########")
    print(dataframe["order_channel"].unique())
    print("########## HOW MANY UNIQUE LAST CHANNEL ##########")
    print(dataframe["last_order_channel"].unique())

variable_analysis(df)

df["total_order_num"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["total_customer_value"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

# Omnichannel check : Omnichannel states that customers shop from both online and offline platforms.
df[(df["order_num_total_ever_online"] > 0) & (df["order_num_total_ever_offline"] > 0)].info()

print("Type of variables ", df.dtypes)

columns_dt = ["first_order_date","last_order_date","last_order_date_online","last_order_date_offline"]
for col in columns_dt:
   df[col] = pd.to_datetime(df[col])

# See the distribution of the number of customers in shopping channels, total number of products purchased and total expenditure.
df.groupby("master_id").agg({"master_id" : lambda x : x.nunique(),
                             "total_order_num" : lambda x : x.sum(),
                              "total_customer_value" : lambda x : x.sum()}).sort_values("total_customer_value", ascending = False)

df.groupby("order_channel").agg({"master_id" : lambda x : x.nunique(),
                             "total_order_num" : lambda x : x.sum(),
                              "total_customer_value" : lambda x : x.sum()}).sort_values("total_customer_value", ascending = False)

df.groupby("order_channel").agg({ "master_id" : "count",
                                 "total_order_num" : ["count","sum","mean"],
                                 "total_customer_value" : ["count","sum","mean"]})

# List the top 10 customers who bring the most profit.
df.groupby("master_id").agg({"total_customer_value" : "sum"}).sort_values("total_customer_value", ascending = False).head(10)

# List the top 10 customers who placed the most orders.
df.groupby("master_id").agg({"total_order_num" : "sum"}).sort_values("total_order_num", ascending = False).head(10)

df.columns[df.columns.str.contains("date")]


# you can define the function that is used for data preparation.
def prep_data(dataframe, head=10):
    dataframe["total_order_num"] = dataframe["order_num_total_ever_online"] + dataframe["order_num_total_ever_offline"]
    dataframe["total_customer_value"] = dataframe["customer_value_total_ever_offline"] + dataframe[
        "customer_value_total_ever_online"]
    date_columns = dataframe.columns[dataframe.columns.str.contains("date")]
    dataframe[date_columns] = dataframe[date_columns].apply(pd.to_datetime)
    return dataframe.head(head)
prep_data(df,10)

# Calculation of RFM Metrics

# Let's take 2 days after the date of the last purchase in the dataset as the analysis date.
df["last_order_date"].max()
analysis_date = dt.datetime(2021,6,1)

(analysis_date - df["last_order_date"]).dt.days

# a new rfm dataframe with customer_id, recency, frequnecy and monetary values
rfm = pd.DataFrame()
rfm["customer_id"] = df["master_id"]
rfm["recency"] = (analysis_date - df["last_order_date"]).dt.days
rfm["frequency"] = df["total_order_num"]
rfm["monetary"] = df["total_customer_value"]
rfm.head()

# Calculating RF and RFM Scores
rfm["recency_score"] = pd.qcut(rfm["recency"], q=5, labels = [5,4,3,2,1])
rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method = "first"), q=5, labels = [1,2,3,4,5])
rfm["monetary_score"] = pd.qcut(rfm["monetary"], q=5, labels = [1,2,3,4,5])
rfm.head()

rfm["RFM_SCORE"] = (rfm["recency_score"].astype(str) + rfm["frequency_score"].astype(str))

rfm.head()

seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}

rfm["segment"] = rfm["RFM_SCORE"].replace(seg_map, regex = True)

rfm[["customer_id","recency_score","frequency_score","monetary_score","RFM_SCORE","segment"]]

rfm[["customer_id","recency_score","frequency_score","monetary_score","RFM_SCORE","segment"]].sort_values("RFM_SCORE",ascending= False).head(10)

#Examine the recency, frequnecy and monetary averages of the segments.
rfm[["segment","recency","frequency","monetary"]].groupby("segment").agg(["mean","count"])

# Case 1: FLO incorporates a new women's footwear brand. The product prices of the brand are above the general customer preferences. For this reason, it is desired to contact the customers who will be interested in the promotion of the brand and product sales. These customers are planned to be loyal and female category shoppers. Save the id numbers of the customers in csv file as new_target_brand_customer_id.cvs
target_segments_customer_ids = rfm[rfm["segment"].isin(["loyal_customers"])]["customer_id"]
customer_ids = df[df["master_id"].isin(target_segments_customer_ids) & df["interested_in_categories_12"].str.contains("KADIN")]["master_id"]
customer_ids.to_csv("new_target_brand_customer.csv", index = False)

# Case 2: A discount of up to 40% is planned for men's and children's products. With this discount, it is desired to specifically target customers who have been good customers in the past but have not been shopping for a long time and new customers who are interested in the relevant categories. Save the ids of the customers in the appropriate profile to csv file as discount_target_ids.csv.
target_customer_ids = rfm[rfm["segment"].isin(["cant_loose","hibernating","new_customers"])]["customer_id"]
cust_id = df[(df["master_id"].isin(target_customer_ids)) & ((df["interested_in_categories_12"].str.contains("ERKEK")) | (df["interested_in_categories_12"].str.contains("ÇOCUK")) )]["master_id"]
cust_id.to_csv("discount_target_ids.csv", index = False)