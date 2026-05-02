###### TEDx-Load-Aggregate-Model + Watch Next
######

import sys

from pyspark.sql import functions as F

from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.dynamicframe import DynamicFrame


# ============================================================
# Percorsi S3 dei CSV
# ============================================================
# Usiamo gli stessi file del job del prof, aggiungendo related_videos.csv
# come sorgente dati per costruire il campo logico watch_next.
# ============================================================

TEDX_DATASET_PATH = "s3://tedx-2026-data-mc/final_list.csv"
DETAILS_DATASET_PATH = "s3://tedx-2026-data-mc/details.csv"
TAGS_DATASET_PATH = "s3://tedx-2026-data-mc/tags.csv"
WATCH_NEXT_DATASET_PATH = "s3://tedx-2026-data-mc/related_videos.csv"


# ============================================================
# Configurazione MongoDB
# ============================================================
# Per il primo test usiamo una collection separata.
# Quando il test è corretto, si può cambiare in "tedx_data"
# oppure in una collection definitiva.
# ============================================================

MONGO_CONNECTION_NAME = "TEDxProf"
MONGO_DATABASE = "unibg_tedx_2026"
MONGO_COLLECTION = "tedx_data"


# ============================================================
# Lettura parametri e inizializzazione Glue/Spark
# ============================================================

args = getResolvedOptions(sys.argv, ["JOB_NAME"])

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session

job = Job(glueContext)
job.init(args["JOB_NAME"], args)

spark.sparkContext.setLogLevel("WARN")


# ============================================================
# Funzioni di supporto
# ============================================================

def read_csv(path):
    """
    Legge un CSV da S3.
    multiLine=true aiuta a gestire campi testuali su più righe,
    ad esempio descrizioni lunghe.
    """
    return (
        spark.read
        .option("header", "true")
        .option("quote", "\"")
        .option("escape", "\"")
        .option("multiLine", "true")
        .csv(path)
    )


def trim_string_columns(df):
    """
    Rimuove spazi iniziali e finali da tutte le colonne testuali.
    """
    for column_name, column_type in df.dtypes:
        if column_type == "string":
            df = df.withColumn(column_name, F.trim(F.col(column_name)))
    return df


def normalize_id(df):
    """
    Normalizza la colonna id convertendola a stringa e rimuovendo spazi.
    """
    if "id" in df.columns:
        df = df.withColumn("id", F.trim(F.col("id").cast("string")))
    return df


def add_missing_columns(df, columns):
    """
    Aggiunge come null eventuali colonne mancanti.
    Serve a rendere il job più robusto se un CSV non contiene un campo atteso.
    """
    for column_name in columns:
        if column_name not in df.columns:
            df = df.withColumn(column_name, F.lit(None).cast("string"))
    return df


def not_empty(column_name):
    """
    Condizione Spark: colonna non nulla e non vuota.
    """
    return F.col(column_name).isNotNull() & (F.trim(F.col(column_name)) != "")


def count_empty(df, column_name):
    """
    Conta quante righe hanno una colonna nulla o vuota.
    """
    return df.filter(~not_empty(column_name)).count()


# ============================================================
# Lettura dei dataset
# ============================================================

tedx_dataset = read_csv(TEDX_DATASET_PATH)
details_dataset = read_csv(DETAILS_DATASET_PATH)
tags_dataset = read_csv(TAGS_DATASET_PATH)
watch_next_dataset = read_csv(WATCH_NEXT_DATASET_PATH)


# ============================================================
# Pulizia iniziale
# ============================================================
# Normalizziamo id e campi testuali.
# Non stiamo ancora aggregando: prepariamo i dati.
# ============================================================

tedx_dataset = normalize_id(trim_string_columns(tedx_dataset))
details_dataset = normalize_id(trim_string_columns(details_dataset))
tags_dataset = normalize_id(trim_string_columns(tags_dataset))
watch_next_dataset = normalize_id(trim_string_columns(watch_next_dataset))


# ============================================================
# Colonne attese
# ============================================================

tedx_dataset = add_missing_columns(
    tedx_dataset,
    ["id", "slug", "speakers", "title", "url"]
)

details_dataset = add_missing_columns(
    details_dataset,
    ["id", "description", "duration", "publishedAt"]
)

tags_dataset = add_missing_columns(
    tags_dataset,
    ["id", "tag"]
)

watch_next_dataset = add_missing_columns(
    watch_next_dataset,
    [
        "id",
        "internalId",
        "related_id",
        "slug",
        "title",
        "duration",
        "viewedCount",
        "presenterDisplayName"
    ]
)


# ============================================================
# Conteggi input
# ============================================================

count_tedx = tedx_dataset.count()
count_details = details_dataset.count()
count_tags = tags_dataset.count()
count_watch_next = watch_next_dataset.count()

print("========== CONTEGGI INPUT ==========")
print(f"final_list.csv       : {count_tedx} righe")
print(f"details.csv          : {count_details} righe")
print(f"tags.csv             : {count_tags} righe")
print(f"related_videos.csv   : {count_watch_next} righe")


# ============================================================
# Controlli qualità sui CSV grezzi
# ============================================================
# Questi controlli misurano anomalie prima dell'aggregazione:
# - righe senza id;
# - righe watch_next senza related_id;
# - duplicati nel dataset principale;
# - duplicati nei tag;
# - duplicati nei suggerimenti watch_next.
# ============================================================

tedx_null_id = count_empty(tedx_dataset, "id")
details_null_id = count_empty(details_dataset, "id")
tags_null_id = count_empty(tags_dataset, "id")
watch_next_null_id = count_empty(watch_next_dataset, "id")
watch_next_null_related_id = count_empty(watch_next_dataset, "related_id")

tedx_valid_rows = tedx_dataset.filter(not_empty("id")).count()
tedx_unique_ids = tedx_dataset.filter(not_empty("id")).select("id").distinct().count()
tedx_duplicate_rows_by_id = tedx_valid_rows - tedx_unique_ids

tags_valid_rows = (
    tags_dataset
    .filter(not_empty("id"))
    .filter(not_empty("tag"))
    .count()
)

tags_unique_rows = (
    tags_dataset
    .filter(not_empty("id"))
    .filter(not_empty("tag"))
    .dropDuplicates(["id", "tag"])
    .count()
)

tags_duplicate_rows = tags_valid_rows - tags_unique_rows

watch_next_valid_rows = (
    watch_next_dataset
    .filter(not_empty("id"))
    .filter(not_empty("related_id"))
    .count()
)

watch_next_unique_rows = (
    watch_next_dataset
    .filter(not_empty("id"))
    .filter(not_empty("related_id"))
    .dropDuplicates(["id", "related_id"])
    .count()
)

watch_next_duplicate_rows = watch_next_valid_rows - watch_next_unique_rows

watch_next_without_title = (
    watch_next_dataset
    .filter(not_empty("id"))
    .filter(not_empty("related_id"))
    .filter(~not_empty("title"))
    .count()
)

print("========== DATA QUALITY - CSV GREZZI ==========")
print(f"final_list righe senza id              : {tedx_null_id}")
print(f"details righe senza id                 : {details_null_id}")
print(f"tags righe senza id                    : {tags_null_id}")
print(f"watch_next righe senza id              : {watch_next_null_id}")
print(f"watch_next righe senza related_id      : {watch_next_null_related_id}")
print(f"final_list duplicati per id            : {tedx_duplicate_rows_by_id}")
print(f"tags duplicati id+tag                  : {tags_duplicate_rows}")
print(f"watch_next duplicati id+related_id     : {watch_next_duplicate_rows}")
print(f"watch_next senza titolo                : {watch_next_without_title}")


# ============================================================
# Dataset principale: final_list.csv
# ============================================================
# Partiamo da final_list.csv.
# Raggruppiamo per id per evitare documenti duplicati nel modello finale.
# Ogni id produce un solo documento MongoDB.
# ============================================================

tedx_base = (
    tedx_dataset
    .filter(not_empty("id"))
    .groupBy("id")
    .agg(
        F.first("slug", ignorenulls=True).alias("slug"),
        F.first("speakers", ignorenulls=True).alias("main_speaker"),
        F.first("title", ignorenulls=True).alias("title"),
        F.first("url", ignorenulls=True).alias("url")
    )
)


# ============================================================
# Details
# ============================================================
# Aggiungiamo descrizione, durata e data di pubblicazione.
# È la stessa logica del job base del prof.
# ============================================================

details_dataset_small = (
    details_dataset
    .filter(not_empty("id"))
    .dropDuplicates(["id"])
    .select(
        F.col("id").alias("id_ref"),
        F.col("description"),
        F.col("duration"),
        F.col("publishedAt")
    )
)

tedx_dataset_main = (
    tedx_base
    .join(
        details_dataset_small,
        tedx_base.id == details_dataset_small.id_ref,
        "left"
    )
    .drop("id_ref")
)

tedx_dataset_main.printSchema()


# ============================================================
# Tags
# ============================================================
# Per ogni talk raccogliamo tutti i tag in un array.
# Rimuoviamo prima eventuali duplicati id+tag.
# ============================================================

tags_dataset_agg = (
    tags_dataset
    .filter(not_empty("id"))
    .filter(not_empty("tag"))
    .withColumn("tag", F.lower(F.trim(F.col("tag"))))
    .dropDuplicates(["id", "tag"])
    .groupBy(F.col("id").alias("id_ref_tags"))
    .agg(
        F.collect_list("tag").alias("tags")
    )
)

tags_dataset_agg.printSchema()

tedx_dataset_agg = (
    tedx_dataset_main
    .join(
        tags_dataset_agg,
        tedx_dataset_main.id == tags_dataset_agg.id_ref_tags,
        "left"
    )
    .drop("id_ref_tags")
)


# ============================================================
# Watch Next
# ============================================================
# La consegna parla di dataset watch_next.
# Nel nostro caso lo costruiamo a partire da related_videos.csv,
# perché quel file contiene già:
# - id = talk corrente;
# - related_id = video suggerito da guardare dopo.
#
# Pulizia:
# - scartiamo righe senza id;
# - scartiamo righe senza related_id;
# - eliminiamo duplicati id+related_id.
#
# Aggregazione:
# - groupBy(id);
# - collect_list dei video suggeriti;
# - creazione array watch_next nel documento finale.
# ============================================================

watch_next_dataset_clean = (
    watch_next_dataset
    .filter(not_empty("id"))
    .filter(not_empty("related_id"))
    .dropDuplicates(["id", "related_id"])
)

watch_next_dataset_agg = (
    watch_next_dataset_clean
    .groupBy(F.col("id").alias("id_ref_watch_next"))
    .agg(
        F.collect_list(
            F.struct(
                F.col("related_id").alias("id"),
                F.col("slug"),
                F.col("title"),
                F.col("duration").cast("double").alias("durationSeconds"),
                F.col("viewedCount").cast("long").alias("viewedCount"),
                F.col("presenterDisplayName")
            )
        ).alias("watch_next")
    )
)

watch_next_dataset_agg.printSchema()

tedx_dataset_total = (
    tedx_dataset_agg
    .join(
        watch_next_dataset_agg,
        tedx_dataset_agg.id == watch_next_dataset_agg.id_ref_watch_next,
        "left"
    )
    .drop("id_ref_watch_next")
)


# ============================================================
# Controllo informativo sui riferimenti Watch Next
# ============================================================
# Verifichiamo quanti related_id non compaiono tra gli id principali.
# Non è un errore bloccante: un video suggerito può puntare a contenuti
# TED/TEDx esterni al catalogo principale caricato nel DWH.
# ============================================================

main_ids = tedx_base.select(F.col("id").alias("main_id"))

watch_next_references = (
    watch_next_dataset_clean
    .select("id", "related_id")
    .dropDuplicates(["id", "related_id"])
)

watch_next_not_in_catalog = (
    watch_next_references
    .join(
        main_ids,
        watch_next_references.related_id == main_ids.main_id,
        "left_anti"
    )
    .count()
)

print("========== DATA QUALITY - WATCH NEXT ==========")
print(f"Watch next senza corrispondenza diretta nel catalogo principale : {watch_next_not_in_catalog}")


# ============================================================
# Documento finale MongoDB-like
# ============================================================
# Come nel job del prof:
# - l'id originale diventa _id;
# - il campo id viene rimosso dal livello principale;
# - tags e watch_next sono array annidati.
# ============================================================

tedx_dataset_total = (
    tedx_dataset_total
    .select(F.col("id").alias("_id"), F.col("*"))
    .drop("id")
)


# ============================================================
# Controlli finali
# ============================================================

total_documents = tedx_dataset_total.count()
unique_ids = tedx_dataset_total.select("_id").distinct().count()
with_tags = tedx_dataset_total.filter(F.col("tags").isNotNull()).count()
with_watch_next = tedx_dataset_total.filter(F.col("watch_next").isNotNull()).count()
without_title = tedx_dataset_total.filter(F.col("title").isNull() | (F.trim(F.col("title")) == "")).count()
without_duration = tedx_dataset_total.filter(F.col("duration").isNull() | (F.trim(F.col("duration")) == "")).count()

print("========== OUTPUT FINALE ==========")
print(f"Documenti finali prodotti : {total_documents}")
print(f"Id unici finali           : {unique_ids}")
print(f"Documenti con tags        : {with_tags}")
print(f"Documenti con watch_next  : {with_watch_next}")
print(f"Documenti senza titolo    : {without_title}")
print(f"Documenti senza durata    : {without_duration}")

if total_documents == 0:
    raise RuntimeError("Errore: il job non ha prodotto documenti finali.")

if total_documents != unique_ids:
    raise RuntimeError("Errore: esistono _id duplicati nel dataset finale.")

print("========== SAMPLE OUTPUT ==========")
tedx_dataset_total.show(5, truncate=False)

print("========== SCHEMA FINALE ==========")
tedx_dataset_total.printSchema()


# ============================================================
# Scrittura su MongoDB
# ============================================================

write_mongo_options = {
    "connectionName": MONGO_CONNECTION_NAME,
    "database": MONGO_DATABASE,
    "collection": MONGO_COLLECTION,
    "ssl": "true",
    "ssl.domain_match": "false"
}

tedx_dataset_dynamic_frame = DynamicFrame.fromDF(
    tedx_dataset_total,
    glueContext,
    "tedx_dataset_with_watch_next"
)

glueContext.write_dynamic_frame.from_options(
    frame=tedx_dataset_dynamic_frame,
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