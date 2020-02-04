#! /bin/bash

hadoop fs -ls -rm /tmp/1970-nixon.txt
hadoop fs -ls -rm -r -f /tmp/nixon_tokens
hadoop fs -ls -rm -r -f /tmp/nixon_token_pos
rm -rf NLTK-demo/nltk_data
rm NLTK-demo/taggers.zip
rm NLTK-demo/tokenizers.zip
