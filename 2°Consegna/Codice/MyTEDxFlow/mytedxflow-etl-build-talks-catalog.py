import sys
import json
import boto3

from awsglue.utils import getResolvedOptions
from awsglue.context import GlueContext
from awsglue.job import Job

from pyspark.context import SparkContext
from pyspark.sql import functions as F


# ============================================================
# Inizializzazione Glue
# ============================================================
# Questa sezione inizializza il contesto AWS Glue e Spark.
# Glue usa Spark per leggere i CSV da S3, trasformarli e creare
# il dataset finale aggregato.
# ============================================================

args = getResolvedOptions(sys.argv, ["JOB_NAME"])

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session

job = Job(glueContext)
job.init(args["JOB_NAME"], args)

spark.sparkContext.setLogLevel("WARN")


# ============================================================
# Percorsi S3 del progetto
# ============================================================
# raw/ contiene i CSV originali.
# processed/ conterrà il file talks.json generato dal job.
# ============================================================

BUCKET_NAME = "mytedxflow-dataset"

RAW_PREFIX = "raw"
PROCESSED_PREFIX = "processed"

RAW_BASE_PATH = f"s3://{BUCKET_NAME}/{RAW_PREFIX}"

FINAL_LIST_PATH = f"{RAW_BASE_PATH}/final_list.csv"
DETAILS_PATH = f"{RAW_BASE_PATH}/details.csv"
TAGS_PATH = f"{RAW_BASE_PATH}/tags.csv"
IMAGES_PATH = f"{RAW_BASE_PATH}/images.csv"
RELATED_VIDEOS_PATH = f"{RAW_BASE_PATH}/related_videos.csv"

OUTPUT_JSON_KEY = f"{PROCESSED_PREFIX}/talks.json"


# ============================================================
# Funzioni di supporto
# ============================================================

def read_csv(path):
    """
    Legge un CSV da S3 usando impostazioni adatte a campi testuali quotati.
    multiLine=true serve perché alcuni campi descrittivi possono contenere
    ritorni a capo o testo complesso.
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
    Questo evita errori dovuti a id, tag o URL scritti con spazi accidentali.
    """
    for column_name, column_type in df.dtypes:
        if column_type == "string":
            df = df.withColumn(column_name, F.trim(F.col(column_name)))
    return df


def normalize_id(df):
    """
    Normalizza la colonna id convertendola a stringa.
    Trattiamo gli id come stringhe perché in MongoDB verranno usati come _id.
    """
    if "id" in df.columns:
        df = df.withColumn("id", F.trim(F.col("id").cast("string")))
    return df


def add_missing_columns(df, columns):
    """
    Aggiunge come null eventuali colonne mancanti.
    Serve per evitare errori se un CSV non contiene un campo atteso.
    """
    for column_name in columns:
        if column_name not in df.columns:
            df = df.withColumn(column_name, F.lit(None).cast("string"))
    return df


def not_empty(column_name):
    """
    Verifica che una colonna non sia nulla e non sia vuota.
    """
    return F.col(column_name).isNotNull() & (F.trim(F.col(column_name)) != "")


def count_empty(df, column_name):
    """
    Conta quante righe hanno una certa colonna nulla o vuota.
    Usata nei controlli di qualità.
    """
    return df.filter(~not_empty(column_name)).count()


def write_json_to_s3(bucket_name, key, data):
    """
    Scrive un unico file JSON su S3.
    Il risultato finale sarà un JSON classico, cioè un array di oggetti talk.
    """
    s3 = boto3.client("s3")

    body = json.dumps(
        data,
        ensure_ascii=False,
        indent=2
    )

    s3.put_object(
        Bucket=bucket_name,
        Key=key,
        Body=body.encode("utf-8"),
        ContentType="application/json"
    )


# ============================================================
# Lettura dei 5 CSV da S3/raw
# ============================================================

final_list = read_csv(FINAL_LIST_PATH)
details = read_csv(DETAILS_PATH)
tags = read_csv(TAGS_PATH)
images = read_csv(IMAGES_PATH)
related_videos = read_csv(RELATED_VIDEOS_PATH)


# ============================================================
# Pulizia iniziale
# ============================================================
# Qui normalizziamo gli id e rimuoviamo spazi inutili.
# Non stiamo ancora aggregando i dati: stiamo solo preparando i CSV.
# ============================================================

final_list = normalize_id(trim_string_columns(final_list))
details = normalize_id(trim_string_columns(details))
tags = normalize_id(trim_string_columns(tags))
images = normalize_id(trim_string_columns(images))
related_videos = normalize_id(trim_string_columns(related_videos))


# Correzione refuso nel dataset: interalId -> internalId
if "interalId" in details.columns:
    details = details.withColumnRenamed("interalId", "internalId")


# ============================================================
# Conteggi input
# ============================================================
# Questi conteggi servono per capire quante righe entrano nel job
# da ciascun CSV originale.
# ============================================================

final_list_count = final_list.count()
details_count = details.count()
tags_count = tags.count()
images_count = images.count()
related_videos_count = related_videos.count()

print("========== CONTEGGI INPUT ==========")
print(f"final_list.csv       : {final_list_count} righe")
print(f"details.csv          : {details_count} righe")
print(f"tags.csv             : {tags_count} righe")
print(f"images.csv           : {images_count} righe")
print(f"related_videos.csv   : {related_videos_count} righe")


# ============================================================
# Controlli qualità iniziali sui CSV grezzi
# ============================================================
# Questa sezione NON modifica ancora i dati finali.
# Serve per capire se nei CSV di partenza ci sono anomalie:
# - righe senza id;
# - id duplicati;
# - duplicati nei dati collegati.
# ============================================================

final_list = add_missing_columns(final_list, ["id", "slug", "speakers", "title", "url"])

details = add_missing_columns(
    details,
    [
        "id",
        "slug",
        "internalId",
        "description",
        "duration",
        "socialDescription",
        "presenterDisplayName",
        "publishedAt"
    ]
)

tags = add_missing_columns(tags, ["id", "tag"])

images = add_missing_columns(images, ["id", "url"])

related_videos = add_missing_columns(
    related_videos,
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

final_list_null_id = count_empty(final_list, "id")
details_null_id = count_empty(details, "id")
tags_null_id = count_empty(tags, "id")
images_null_id = count_empty(images, "id")
related_videos_null_id = count_empty(related_videos, "id")

final_list_unique_ids = final_list.filter(not_empty("id")).select("id").distinct().count()
details_unique_ids = details.filter(not_empty("id")).select("id").distinct().count()

final_list_duplicate_rows_by_id = final_list.filter(not_empty("id")).count() - final_list_unique_ids
details_duplicate_rows_by_id = details.filter(not_empty("id")).count() - details_unique_ids

tags_duplicate_rows = (
    tags
    .filter(not_empty("id"))
    .filter(not_empty("tag"))
    .count()
    -
    tags
    .filter(not_empty("id"))
    .filter(not_empty("tag"))
    .dropDuplicates(["id", "tag"])
    .count()
)

images_duplicate_rows = (
    images
    .filter(not_empty("id"))
    .filter(not_empty("url"))
    .count()
    -
    images
    .filter(not_empty("id"))
    .filter(not_empty("url"))
    .dropDuplicates(["id", "url"])
    .count()
)

related_duplicate_rows = (
    related_videos
    .filter(not_empty("id"))
    .filter(not_empty("related_id"))
    .count()
    -
    related_videos
    .filter(not_empty("id"))
    .filter(not_empty("related_id"))
    .dropDuplicates(["id", "related_id"])
    .count()
)

print("========== DATA QUALITY - CSV GREZZI ==========")
print(f"final_list righe senza id              : {final_list_null_id}")
print(f"details righe senza id                 : {details_null_id}")
print(f"tags righe senza id                    : {tags_null_id}")
print(f"images righe senza id                  : {images_null_id}")
print(f"related_videos righe senza id          : {related_videos_null_id}")
print(f"final_list duplicati per id            : {final_list_duplicate_rows_by_id}")
print(f"details duplicati per id               : {details_duplicate_rows_by_id}")
print(f"tags duplicati id+tag                  : {tags_duplicate_rows}")
print(f"images duplicati id+url                : {images_duplicate_rows}")
print(f"related_videos duplicati id+related_id : {related_duplicate_rows}")


# ============================================================
# Base principale: final_list.csv
# ============================================================
# final_list è usato come dataset principale.
# Raggruppiamo per id così ogni talk principale compare una volta sola.
# Questo elimina duplicati a livello di talk.
# ============================================================

final_base = (
    final_list
    .filter(not_empty("id"))
    .groupBy("id")
    .agg(
        F.first("slug", ignorenulls=True).alias("slugFinal"),
        F.first("speakers", ignorenulls=True).alias("speakers"),
        F.first("title", ignorenulls=True).alias("title"),
        F.first("url", ignorenulls=True).alias("url")
    )
)


# ============================================================
# Details: metadati descrittivi del talk
# ============================================================
# Da details prendiamo descrizione, durata, data pubblicazione,
# speaker e internalId. Usiamo dropDuplicates(["id"]) perché vogliamo
# un solo record details per ogni talk.
# ============================================================

details_small = (
    details
    .filter(not_empty("id"))
    .dropDuplicates(["id"])
    .select(
        F.col("id").alias("id_ref"),
        F.col("slug").alias("slugDetails"),
        F.col("internalId"),
        F.col("description"),
        F.col("duration"),
        F.col("socialDescription"),
        F.col("presenterDisplayName"),
        F.col("publishedAt")
    )
)


# ============================================================
# Tags: più righe dello stesso talk diventano un array tags
# ============================================================
# Rimuoviamo tag vuoti, normalizziamo in minuscolo e togliamo
# duplicati id+tag. collect_set garantisce tag unici per talk.
# ============================================================

tags_grouped = (
    tags
    .filter(not_empty("id"))
    .filter(not_empty("tag"))
    .withColumn("tag", F.lower(F.trim(F.col("tag"))))
    .dropDuplicates(["id", "tag"])
    .groupBy(F.col("id").alias("id_ref"))
    .agg(
        F.collect_set("tag").alias("tags")
    )
)


# ============================================================
# Images: più immagini dello stesso talk diventano un array images
# ============================================================
# Rimuoviamo immagini senza id o senza URL e togliamo duplicati
# sulla coppia id+url.
# ============================================================

images_grouped = (
    images
    .filter(not_empty("id"))
    .filter(not_empty("url"))
    .dropDuplicates(["id", "url"])
    .groupBy(F.col("id").alias("id_ref"))
    .agg(
        F.collect_list(
            F.struct(
                F.col("url").alias("url")
            )
        ).alias("images")
    )
)


# ============================================================
# Related videos: più related videos diventano un array relatedVideos
# ============================================================
# Rimuoviamo righe senza id principale o senza related_id.
# Poi togliamo duplicati sulla coppia id+related_id.
# Ogni elemento dell'array rappresenta un video correlato.
# ============================================================

related_grouped = (
    related_videos
    .filter(not_empty("id"))
    .filter(not_empty("related_id"))
    .dropDuplicates(["id", "related_id"])
    .groupBy(F.col("id").alias("id_ref"))
    .agg(
        F.collect_list(
            F.struct(
                F.col("related_id").alias("id"),
                F.col("slug").alias("slug"),
                F.col("title").alias("title"),
                F.col("duration").cast("double").alias("durationSeconds"),
                F.col("viewedCount").cast("long").alias("viewedCount"),
                F.col("presenterDisplayName").alias("presenterDisplayName")
            )
        ).alias("relatedVideos")
    )
)


# ============================================================
# Controllo qualità sui related videos
# ============================================================
# Verifichiamo se i related_id presenti nel CSV dei video correlati
# hanno una corrispondenza diretta nel catalogo principale.
# Questo controllo è solo informativo: i related videos possono
# puntare anche a contenuti TED/TEDx esterni al catalogo principale.
# ============================================================

main_ids = final_base.select(F.col("id").alias("main_id"))

related_references = (
    related_videos
    .filter(not_empty("id"))
    .filter(not_empty("related_id"))
    .select("id", "related_id")
    .dropDuplicates(["id", "related_id"])
)

related_not_in_catalog = (
    related_references
    .join(main_ids, related_references.related_id == main_ids.main_id, "left_anti")
    .count()
)

related_without_title = related_videos.filter(
    not_empty("id") &
    not_empty("related_id") &
    (~not_empty("title"))
).count()

print("========== DATA QUALITY - RELATED VIDEOS ==========")
print(f"Related videos senza corrispondenza diretta nel catalogo principale : {related_not_in_catalog}")
print(f"Related videos senza titolo                                        : {related_without_title}")


# ============================================================
# Join finale tramite id
# ============================================================
# Costruiamo il documento finale del talk unendo:
# - base principale;
# - dettagli;
# - tag aggregati;
# - immagini aggregate;
# - related videos aggregati.
# ============================================================

talks = (
    final_base
    .join(details_small, final_base.id == details_small.id_ref, "left")
    .drop("id_ref")
)

talks = (
    talks
    .join(tags_grouped, talks.id == tags_grouped.id_ref, "left")
    .drop("id_ref")
)

talks = (
    talks
    .join(images_grouped, talks.id == images_grouped.id_ref, "left")
    .drop("id_ref")
)

talks = (
    talks
    .join(related_grouped, talks.id == related_grouped.id_ref, "left")
    .drop("id_ref")
)


# ============================================================
# Campi finali utili all'app
# ============================================================
# Creiamo campi più comodi per l'app:
# - slug finale;
# - speaker leggibile;
# - durata in secondi e minuti;
# - searchText per ricerca testuale semplice.
# ============================================================

talks = talks.withColumn(
    "slug",
    F.coalesce(F.col("slugDetails"), F.col("slugFinal"))
)

talks = talks.withColumn(
    "speaker",
    F.coalesce(
        F.col("presenterDisplayName"),
        F.col("speakers"),
        F.lit("Speaker non disponibile")
    )
)

talks = talks.withColumn(
    "durationSeconds",
    F.col("duration").cast("double")
)

talks = talks.withColumn(
    "durationMinutes",
    F.round(F.col("durationSeconds") / F.lit(60), 1)
)

talks = talks.withColumn(
    "searchText",
    F.lower(
        F.concat_ws(
            " ",
            F.col("title"),
            F.col("speaker"),
            F.col("description"),
            F.col("socialDescription"),
            F.array_join(F.col("tags"), " ")
        )
    )
)


# ============================================================
# Documento finale: una riga Spark = un talk completo
# ============================================================
# Manteniamo sia _id sia id:
# - _id serve a MongoDB come identificativo tecnico univoco;
# - id resta utile all'app e agli eventuali riferimenti applicativi.
# ============================================================

talks_final = talks.select(
    F.col("id").alias("_id"),
    F.col("id").alias("id"),
    F.col("slug"),
    F.col("title"),
    F.col("url"),
    F.col("speaker"),
    F.col("description"),
    F.col("socialDescription"),
    F.col("durationSeconds"),
    F.col("durationMinutes"),
    F.col("publishedAt"),
    F.col("internalId"),
    F.col("tags"),
    F.col("images"),
    F.col("relatedVideos"),
    F.col("searchText")
)


# ============================================================
# Controlli qualità sul dataset finale
# ============================================================
# Questi controlli verificano se il dataset finale è coerente
# per essere usato dall'app e caricato su MongoDB.
# Non eliminiamo automaticamente i record anomali: li segnaliamo.
# ============================================================

total_documents = talks_final.count()
unique_ids = talks_final.select("_id").distinct().count()

with_tags = talks_final.filter(F.col("tags").isNotNull()).count()
with_images = talks_final.filter(F.col("images").isNotNull()).count()
with_related = talks_final.filter(F.col("relatedVideos").isNotNull()).count()

without_id = talks_final.filter(F.col("_id").isNull() | (F.trim(F.col("_id")) == "")).count()
without_title = talks_final.filter(F.col("title").isNull() | (F.trim(F.col("title")) == "")).count()
without_speaker = talks_final.filter(F.col("speaker").isNull() | (F.trim(F.col("speaker")) == "")).count()
without_description = talks_final.filter(F.col("description").isNull() | (F.trim(F.col("description")) == "")).count()
without_url = talks_final.filter(F.col("url").isNull() | (F.trim(F.col("url")) == "")).count()

duration_null = talks_final.filter(F.col("durationSeconds").isNull()).count()
duration_zero_or_negative = talks_final.filter(
    F.col("durationSeconds").isNotNull() &
    (F.col("durationSeconds") <= 0)
).count()

invalid_talk_url = talks_final.filter(
    F.col("url").isNotNull() &
    (~F.lower(F.col("url")).rlike(r"^https?://(www\.)?ted\.com/talks/"))
).count()

invalid_image_url = (
    images
    .filter(not_empty("id"))
    .filter(not_empty("url"))
    .filter(~F.lower(F.col("url")).rlike(r"^https?://"))
    .count()
)

print("========== OUTPUT FINALE ==========")
print(f"Documenti finali prodotti : {total_documents}")
print(f"Id unici finali           : {unique_ids}")
print(f"Talk con tags             : {with_tags}")
print(f"Talk con immagini         : {with_images}")
print(f"Talk con related videos   : {with_related}")

print("========== DATA QUALITY - DATASET FINALE ==========")
print(f"Talk senza _id                         : {without_id}")
print(f"Talk senza titolo                      : {without_title}")
print(f"Talk senza speaker                     : {without_speaker}")
print(f"Talk senza descrizione                 : {without_description}")
print(f"Talk senza URL                         : {without_url}")
print(f"Talk con durata nulla/non numerica     : {duration_null}")
print(f"Talk con durata <= 0                   : {duration_zero_or_negative}")
print(f"Talk con URL non TED talks             : {invalid_talk_url}")
print(f"Immagini con URL non http/https        : {invalid_image_url}")


# ============================================================
# Controlli bloccanti minimi
# ============================================================
# Questi controlli bloccano il job solo se il risultato finale
# non è caricabile correttamente su MongoDB:
# - nessun documento prodotto;
# - _id mancanti;
# - _id duplicati.
# Le altre anomalie vengono solo segnalate nei log.
# ============================================================

if total_documents == 0:
    raise RuntimeError("Errore: il job non ha prodotto documenti finali.")

if without_id > 0:
    raise RuntimeError("Errore: alcuni documenti finali non hanno _id.")

if total_documents != unique_ids:
    raise RuntimeError("Errore: esistono _id duplicati nel dataset finale.")


print("========== SAMPLE OUTPUT ==========")
talks_final.show(5, truncate=False)
talks_final.printSchema()


# ============================================================
# Conversione in JSON classico
# ============================================================
# Il dataset è piccolo, circa 7055 talk, quindi possiamo raccogliere
# i record sul driver e scrivere un unico file JSON classico.
# Il file finale sarà un array di oggetti talk.
# ============================================================

talks_json_records = [
    json.loads(row)
    for row in talks_final.toJSON().collect()
]


# ============================================================
# Normalizzazione finale lato Python
# ============================================================
# Se alcuni array sono null, li trasformiamo in array vuoti.
# Questo rende più semplice l'uso lato app, perché tags/images/relatedVideos
# saranno sempre array.
# ============================================================

for talk in talks_json_records:
    if talk.get("tags") is None:
        talk["tags"] = []

    if talk.get("images") is None:
        talk["images"] = []

    if talk.get("relatedVideos") is None:
        talk["relatedVideos"] = []


# ============================================================
# Scrittura unico output JSON su S3/processed
# ============================================================
# Il job sovrascrive processed/talks.json con la nuova versione
# del dataset aggregato e controllato.
# ============================================================

write_json_to_s3(BUCKET_NAME, OUTPUT_JSON_KEY, talks_json_records)

print("========== FILE SCRITTO SU S3 ==========")
print(f"JSON finale : s3://{BUCKET_NAME}/{OUTPUT_JSON_KEY}")


# ============================================================
# Chiusura job
# ============================================================

job.commit()