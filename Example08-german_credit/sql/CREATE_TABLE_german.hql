-- remove & add tables for german data 
-- requires german data to already be loaded in hdfs

CREATE EXTERNAL TABLE IF NOT EXISTS default.german (
    cred        INT COMMENT "0  response   - Creditability",
    acct_bal    INT COMMENT "1  cat(1,4)   - Account Balance",
    dur_cred    INT COMMENT "2  cont       - Duration of Credit (month)",
    pay_stat    INT COMMENT "3  cat(0,4)   - Payment Status of Previous Credit",
    purpose     INT COMMENT "4  cat(0,10)  - Purpose",
    cred_amt    INT COMMENT "5  cont       - Credit Amount",
    value       INT COMMENT "6  cat(1,5)   - Value Savings/Stocks",
    len_emp     INT COMMENT "7  cat(1,5)   - Length of current employment",
    install_pc  INT COMMENT "8  cat(1,4)   - Instalment per cent",
    sex_married INT COMMENT "9  cat(1,4)   - Sex & Marital Status",
    guarantors  INT COMMENT "10 cat(1,3)   - Guarantors",
    dur_addr    INT COMMENT "11 cat(1,4)   - Duration in Current address",
    max_val     INT COMMENT "12 cat(1,4)   - Most valuable available asset",
    age         INT COMMENT "13 cont - Age (years)",
    concurr     INT COMMENT "14 cat(1,3)  - Concurrent Credits",
    typ_aprtmnt INT COMMENT "15 cat(1,3)  - Type of apartment",
    no_creds    INT COMMENT "16 cat(1,4)  - No of Credits at this Bank",
    occupation  INT COMMENT "17 cat(1,4)  - Occupation",
    no_dep      INT COMMENT "18 cat(1,2)  - No of dependents",
    telephone   INT COMMENT "19 cat(1,2)  - Telephone",
    foreign_wkr INT COMMENT "20 cat(1,2)  - Foreign Worker")
ROW FORMAT DELIMITED FIELDS TERMINATED BY ','
LINES TERMINATED BY '\n' 
--location 'hdfs://nameservice1/user/hive/warehouse/german_credit'
TBLPROPERTIES ("skip.header.line.count"="1");

DROP TABLE IF EXISTS default.german_parquet;

CREATE EXTERNAL TABLE default.german_parquet (
    cred        INT COMMENT "0  response   - Creditability",
    acct_bal    INT COMMENT "1  cat(1,4)   - Account Balance",
    dur_cred    INT COMMENT "2  cont       - Duration of Credit (month)",
    pay_stat    INT COMMENT "3  cat(0,4)   - Payment Status of Previous Credit",
    purpose     INT COMMENT "4  cat(0,10)  - Purpose",
    cred_amt    INT COMMENT "5  cont       - Credit Amount",
    value       INT COMMENT "6  cat(1,5)   - Value Savings/Stocks",
    len_emp     INT COMMENT "7  cat(1,5)   - Length of current employment",
    install_pc  INT COMMENT "8  cat(1,4)   - Instalment per cent",
    sex_married INT COMMENT "9  cat(1,4)   - Sex & Marital Status",
    guarantors  INT COMMENT "10 cat(1,3)   - Guarantors",
    dur_addr    INT COMMENT "11 cat(1,4)   - Duration in Current address",
    max_val     INT COMMENT "12 cat(1,4)   - Most valuable available asset",
    age         INT COMMENT "13 cont - Age (years)",
    concurr     INT COMMENT "14 cat(1,3)  - Concurrent Credits",
    typ_aprtmnt INT COMMENT "15 cat(1,3)  - Type of apartment",
    no_creds    INT COMMENT "16 cat(1,4)  - No of Credits at this Bank",
    occupation  INT COMMENT "17 cat(1,4)  - Occupation",
    no_dep      INT COMMENT "18 cat(1,2)  - No of dependents",
    telephone   INT COMMENT "19 cat(1,2)  - Telephone",
    foreign_wkr INT COMMENT "20 cat(1,2)  - Foreign Worker")
  ROW FORMAT SERDE 'parquet.hive.serde.ParquetHiveSerDe'
  STORED AS
    INPUTFORMAT "parquet.hive.DeprecatedParquetInputFormat"
    OUTPUTFORMAT "parquet.hive.DeprecatedParquetOutputFormat"
--LOCATION 'hdfs://nameservice1/user/hive/warehouse/german_parquet'
    ;

set parquet.compression=SNAPPY;
INSERT OVERWRITE TABLE default.german_parquet 
SELECT * FROM default.german;