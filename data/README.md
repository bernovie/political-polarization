# Important Columns

* Input.article_id | uniquely identifies an article for external linking.
* Answer.author | the label given to the author stance 
* Answer.fiscal_topic_{x} | the label given to the x paragraph in the fiscal dimension
    - moderate_val_left: left leaning
    - moderate_val_right: right leaning
    - mixed_val: paragraph contains both left and right leaning elements
    - default: paragraph is not relevant to dimension
* Answer.social_topic_{x} | the label given to the x paragraph in the social dimension
* Answer.foreign_topic_{x} | the label given to the x paragraph in the foreign dimension

You can mostly ignore the "Answer.*_topic_{x}_checkbox.on" columns

You can mostly ignore the "Answer.line_num_*_topic_{x}" columns

# Metadata

To get metadata such as the news source (E.g. New York Times, Christian Science Monitor) or original text use the article id to link the entry with the file "topic_34k_original_and_segmented_articles_with_metadata.csv"