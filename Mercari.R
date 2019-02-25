library(dplyr)
library(data.table)

# Reading the data.

system.time(
train <- as_tibble(fread('../input/train.tsv', encoding="UTF-8"))
)
system.time(
test <- as_tibble(fread('../input/test.tsv', encoding="UTF-8"))
)

# Are there any rows with the zero price?

sum(train$price == 0)

# Yes! Let's remove them.

train <- train[train$price != 0, ]

# Let's introduce a common name for the id columns of the data sets.

colnames(train)[1] <- 'id'
colnames(test)[1] <- 'id'

# Computing the log-price

train$log_price <- log(train$price + 1)

# Add 'price' and 'log_price' columns filled with NA's 
# to the test data set.

test$price <- rep(NA, nrow(test))
test$log_price <- rep(NA, nrow(test))

# Binding together the training and test data. 

data <- rbind(train, test)

# Storing the number of rows in the training data set. 
# We'll use it later to separate the training and test data.

size_train <- nrow(train)

# Remove the train and test data set to free up some memory.

rm(train, test)

# The 'item_description' columns contains entries 'No description yet'. 
# Let's convert it to a single word.

library(stringr)
data <- data %>%
    mutate(item_description = 
        if_else(item_description == '' | 
            str_to_lower(item_description) == 'no description yet', 
            'nodescriptiongiven', item_description))

# Observe that the entries of the 'category_name' column are separated
# into subcategories by the '/' symbol. 

# How many subcategories are there? To find out, we can run
#
# table(sapply(data$category_name, str_count, '/'))
#
# We won't do it here to save time but if you actually do that you
# will see that the maximum number of the '/' symbols is 4. Therefore
# the maximum number of subcategories is 5. Also, observe that 9385
# entries do not have the '/' symbol. Are they empty? Let's count
# the number of blank entries in the 'category_name' column.

empty_category <- data$category_name == ""
sum(empty_category)
rm(empty_category)

# Yes! 9385 entries are blank. 

# Put the five subcategories into individual columns.

sub_cat_names <- c('sub_cat_1', 'sub_cat_2', 'sub_cat_3', 
                   'sub_cat_4',  'sub_cat_5')
                   
print('Spliting the category names...')

system.time(
    data[sub_cat_names] <- tstrsplit(data$category_name, split = "/")
)

# Filling empty entries in the data set

data[data$brand_name == '', 'brand_name'] <- 'Unknown'

for (name in sub_cat_names) {
    data[is.na(data[name]), name] <- 'Unknown'
}

# The names of the variables to be included in the model

cols <- c('item_condition_id', 'brand_name', 'shipping', sub_cat_names)

# Convert the variable to factors

data[cols] <- as.data.frame(apply(data[cols], 2, factor))

library(quanteda)

# Define a new function to produce data-frequency matricies
# for n-grams.

build_dfm <- function(x, n = 1, min = 2500){
# Make the document-frequency matrix using the 'quanteda' package.
                mat <- dfm(x, tolower = TRUE, remove_punct = TRUE, 
                       remove_symbols = TRUE, remove_numbers = TRUE, 
                       remove = stopwords("english"), ngrams = n)
# Trimming the matrix
mat <- dfm_trim(mat, min_count = min)
# Applying tf-idf
mat <- tfidf(mat)
return(mat)
            }

# Processing the 'item_description' text column.

print('Processing the "item_description" text column...')

system.time(
    dfm_item_description <- build_dfm(x = data$item_description,
                                      n = 1, min = 2500)
)

# Processing the 'name' text column.

print('Processing the "name" text column...')

system.time(
    dfm_name <- build_dfm(x = data$name, n = 1, min = 100)
)

# Preparing the model matrix.

library(Matrix)

print('Preparing the sparse model matrix...')

system.time(
    data_sparse <- sparse.model.matrix(
    ~ item_condition_id + brand_name + shipping +
    sub_cat_1 + sub_cat_2 + sub_cat_3 + sub_cat_4 + sub_cat_5,
    data = data[cols]
    )
)

## Fix for cbind dfm and sparse matrix.

system.time(
    class(dfm_item_description) <- class(data_sparse)
)

system.time(
    class(dfm_name) <- class(data_sparse)
)

system.time(
    data_sparse <- cbind(
        data_sparse, 
        dfm_item_description,
        dfm_name)
    )

rownames(data_sparse) <- NULL

# Extracting the dependent variable from 'data'

outcomes <- as.numeric(data$log_price[1:size_train])

library(xgboost)

# The training/testing data for XGBoost.

print('Preparing the model matrix for XGBoost...')

dtrain <- xgb.DMatrix(data_sparse[1:size_train, ], label = outcomes)

dtest <- xgb.DMatrix(data_sparse[(size_train + 1) : nrow(data_sparse), ])

# Fitting the XGBoost model.

print('Fitting with XGBoost...')

system.time(
    model <- xgboost(data = dtrain, nround = 1050, objective = "reg:linear")
)
print('Preparing the data for submission...')

log_price <- predict(model, dtest)
price <- exp(log_price) - 1

submission <- data.frame(
    test_id = as.integer(seq(0, nrow(dtest)-1)),
    price = price
)

write.csv(submission, 
          file = "Mercary_submission.csv", 
          row.names = FALSE
)