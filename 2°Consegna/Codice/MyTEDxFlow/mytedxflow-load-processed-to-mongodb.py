import sys

from awsglue.utils import getResolvedOptions
from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.dynamicframe import DynamicFrame

from pyspark.context import SparkContext
from pyspark.sql import functions as F


# ============================================================
# Inizializzazione Glue
# ============================================================

args = getResolvedOptions(sys.argv, ["JOB_NAME"])

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session

job = Job(glueContext)
job.init(args["JOB_NAME"], args)

spark.sparkContext.setLogLevel("WARN")


# ============================================================
# Percorso S3 del file processato
# ============================================================

PROCESSED_TALKS_PATH = "s3://mytedxflow-dataset/processed/talks.json"


# ============================================================
# Configurazione MongoDB
# ============================================================

MONGO_CONNECTION_NAME = "TEDxProf"
MONGO_DATABASE = "mytedxflow"
MONGO_COLLECTION = "talks"


# ============================================================
# Lettura del file JSON processato
# ============================================================
# talks.json è un JSON classico:
# [
#   { talk 1 },
#   { talk 2 },
#   ...
# ]
#
# Per questo usiamo multiLine = true.
# ============================================================

talks_df = (
    spark.read
    .option("multiLine", "true")
    .json(PROCESSED_TALKS_PATH)
)


# ============================================================
# Controlli sui dati letti
# ============================================================

total_documents = talks_df.count()
unique_ids = talks_df.select("_id").distinct().count()
without_id = talks_df.filter(F.col("_id").isNull() | (F.trim(F.col("_id")) == "")).count()
without_title = talks_df.filter(F.col("title").isNull() | (F.trim(F.col("title")) == "")).count()

print("========== LETTURA FILE PROCESSED ==========")
print(f"File letto da: {PROCESSED_TALKS_PATH}")
print(f"Documenti letti: {total_documents}")
print(f"Id unici: {unique_ids}")
print(f"Documenti senza _id: {without_id}")
print(f"Documenti senza title: {without_title}")

print("========== SCHEMA DATI ==========")
talks_df.printSchema()

print("========== SAMPLE DATI ==========")
talks_df.show(5, truncate=False)


# ============================================================
# Controllo minimo prima della scrittura
# ============================================================

if total_documents == 0:
    raise RuntimeError("Errore: il file processed/talks.json è stato letto ma non contiene documenti.")

if without_id > 0:
    raise RuntimeError("Errore: alcuni documenti non hanno il campo _id. MongoDB richiede un identificativo valido.")

if total_documents != unique_ids:
    raise RuntimeError("Errore: esistono _id duplicati nel dataset. La scrittura su MongoDB fallirebbe o genererebbe dati ambigui.")


# ============================================================
# Conversione Spark DataFrame -> Glue DynamicFrame
# ============================================================

talks_dynamic_frame = DynamicFrame.fromDF(
    talks_df,
    glueContext,
    "talks_dynamic_frame"
)


# ============================================================
# Scrittura su MongoDB Atlas
# ============================================================

write_mongo_options = {
    "connectionName": MONGO_CONNECTION_NAME,
    "database": MONGO_DATABASE,
    "collection": MONGO_COLLECTION,
    "ssl": "true",
    "ssl.domain_match": "false"
}

glueContext.write_dynamic_frame.from_options(
    frame=talks_dynamic_frame,
    connection_type="mongodb",
    connection_options=write_mongo_options
)


print("========== SCRITTURA MONGODB COMPLETATA ==========")
print(f"Connection: {MONGO_CONNECTION_NAME}")
print(f"Database: {MONGO_DATABASE}")
print(f"Collection: {MONGO_COLLECTION}")
print(f"Documenti caricati: {total_documents}")


# ============================================================
# Chiusura job
# ============================================================

job.commit()