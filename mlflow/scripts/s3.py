from dagshub import get_repo_bucket_client
# Get a boto3.client object
s3 = get_repo_bucket_client("sarahlunette/Intro_MLOPS_tsunamis")

# Upload file
s3.upload_file(
    Bucket="Intro_MLOPS_tsunamis",  # name of the repo
    Filename="../../data/processed/human_damages.csv",  # local path of file to upload
    Key="human_damages.csv",  # remote path where to upload the file
)
# Upload file
s3.upload_file(
    Bucket="Intro_MLOPS_tsunamis",  # name of the repo
    Filename="../../data/raw/countries-by-population-density-_-countries-by-density-2024.csv",  # local path of file to upload
    Key="countries-by-population-density-_-countries-by-density-2024.csv",  # remote path where to upload the file
)
# Upload file
s3.upload_file(
    Bucket="Intro_MLOPS_tsunamis",  # name of the repo
    Filename="../../data/raw/gdp_per_capita.csv",  # local path of file to upload
    Key="gdp_per_capita.csv",  # remote path where to upload the file
)
# Upload file
s3.upload_file(
    Bucket="Intro_MLOPS_tsunamis",  # name of the repo
    Filename="../../data/raw/Historical_Tsunami_Event_Locations_with_Runups.csv",  # local path of file to upload
    Key="Historical_Tsunami_Event_Locations_with_Runups.csv",  # remote path where to upload the file
)
