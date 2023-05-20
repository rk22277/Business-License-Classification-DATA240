import pandas as pd
import category_encoders as ce
import warnings
warnings.filterwarnings("ignore")

# change the columns name
def replace_char(df, to_replace, replace_with):
    new_col_name = [col.replace(to_replace,replace_with).lower() for col in df.columns]
    return new_col_name

# Function to drop columns from data
def drop_col(df, col_list):
    for col in col_list:
        if col not in df.columns:
            raise KeyError(
                f"{col} does not exit in dataframe")
    df=df.drop(col_list, axis=1)
    return(df)

# Function to convert string object into date
def convert_to_date(col):
    new_col = pd.DatetimeIndex(col)
    return new_col

# Function to find no.of days between different application status date
def date_diff(col_1, col_2):
    diff_col = (col_1 - col_2).dt.days
    return diff_col

# Function to impute missing values
def impute(df, value=0):
    df = df.fillna(value)
    return df

# Function to encode categorical variables
def convert_numeric(col):
    conv_col=col.apply(lambda col:pd.Categorical(col).codes)
    return conv_col


# does target encoding for the column listed
def target_encoding(df,col_to_transform):
    """This function target encodes the feature and returns the dataframe

    Args:
        df(pandas dataframe) : dataframe to be transformed  
        col_to_transform (List): list of column to be transformed 
    """
    enc=ce.OneHotEncoder().fit(df.target.astype(str))
    y_onehot=enc.transform(df.target.astype(str))
    class_names=y_onehot.columns
    for class_ in class_names:
        enc=ce.TargetEncoder(smoothing=0)
        temp = enc.fit_transform(df[col_to_transform],y_onehot[class_])
        temp.columns=[str(x)+'_'+str(class_) for x in temp.columns]
        df = pd.concat([df,temp],axis=1)
    return df

def random_sampling(df,target,target_prop):
    """Function to generate random sampling based on the target proportion

    Args:
        df(pandas dataframe) : dataframe to be transformed  
        target (List): Distinct value in the target column
        target_prop (List): Fraction of proportion to be present after the sampling process
    """
    df_list = []
    for i in range(len(target)):
        if target_prop[i] > 1:
            temp_df = df[df.target==target[i]].sample(frac=target_prop[i],replace=True)
        else:
            temp_df = df[df.target==target[i]].sample(frac=target_prop[i],replace=False)
        df_list.append(temp_df)
    final_df = pd.concat(df_list)
    return final_df

#FUNCTION TO IDENTIFY OUTLIERS USING IQR METHOD
def iqr(col):
    q1 = col.quantile(0.25)
    q3 = col.quantile(0.75)
    iqr = q3 - q1
    min_ = q1 - 1.5 * iqr
    max_ = q3 + 1.5 * iqr
    return col[((col < min_) | (col > max_))]

#FUNCTION TO DETECT OUTLIERS USING Z-SCORE METHOD
def zscore_outlier(col,lb,ub):
    zscore = ((col - col.mean()) / col.std()).copy()
    return col[((zscore < lb) | (zscore > ub))]    