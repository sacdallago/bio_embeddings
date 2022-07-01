import logging
from typing import List

import gridfs
import pymongo
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database

from webserver.utilities.configuration import configuration

logger = logging.getLogger(__name__)

ten_days = 10 * 24 * 60 * 60

# Because entries in capped collections in mongodb cannot be resized, all status names need to have the same size.
# That's why we define these constants to be used as attributes in the database:
JOB_PENDING = "pend"
JOB_DONE = "done"
JOB_FAILED = "fail"


_client = MongoClient(configuration["web"]["mongo_url"])
_response_cache_db: Database = _client.response_cache
_collection: Database = _client.file_storage
_fs = gridfs.GridFS(_collection)
_collection.fs.files.create_index("uploadDate", expireAfterSeconds=ten_days)


def get_or_create_cache(name: str) -> Collection:
    # Cap the collections at 10GB (which should never be reached anyway)
    if name not in _response_cache_db.list_collection_names():
        logger.info("Creating mongodb response_cache collection")
        cache_collection = _response_cache_db.create_collection(name, size=10 * 1024 * 1024 * 1024, capped=True)
    else:
        cache_collection = _response_cache_db[name]
    # We use `uploadDate` for consistency with gridFS
    cache_collection.create_index("uploadDate", expireAfterSeconds=ten_days)
    return cache_collection


# Caches for the direct feature extractors
get_embedding_cache = get_or_create_cache("get_embedding_cache")
get_features_cache = get_or_create_cache("get_features_cache")
get_residue_landscape_cache = get_or_create_cache("get_residue_landscape_cache")
get_structure_cache = get_or_create_cache("get_structure_cache")
get_structure_jobs = get_or_create_cache("get_structure_jobs")

# Indexes, otherwise it gets really slow with some GB of data
if "model_name_sequence" not in get_features_cache.index_information():
    get_features_cache.create_index(
        ([("model_name", pymongo.ASCENDING), ("sequence", pymongo.HASHED)]),
        name="model_name_sequence",
    )
if "model_name_sequence" not in get_embedding_cache.index_information():
    get_embedding_cache.create_index(
        ([("model_name", pymongo.ASCENDING), ("sequence", pymongo.HASHED)]),
        name="model_name_sequence",
    )
if "structure_sequence" not in get_structure_cache.index_information():
    get_structure_cache.create_index(
        ([("predictor_name", pymongo.ASCENDING), ("sequence", pymongo.HASHED)]),
        name="structure_sequence",
    )
if "structure_sequence_jobs" not in get_structure_jobs.index_information():
    get_structure_jobs.create_index(
        ([("predictor_name", pymongo.ASCENDING), ("sequence", pymongo.HASHED)]),
        name="structure_sequence_jobs",
    )

if "model_name_sequence" not in get_residue_landscape_cache.index_information():
    get_residue_landscape_cache.create_index(
        ([("model_name", pymongo.ASCENDING), ("sequence", pymongo.HASHED)]),
        name="model_name_sequence",
    )


def write_file(job_id, file_identifier, file_path):
    with open(file_path, "rb") as f:
        index = _fs.put(f, filename=file_identifier, job=job_id)

    return index


def get_file(job_id, file_identifier):
    db_file = _fs.find_one(filter={
        "filename": file_identifier,
        "job": job_id
    })

    if db_file:
        return _fs.get(db_file._id)
    else:
        return None


def get_list_of_files(job_id) -> List:
    file_list = _fs.find(filter={
        "job": job_id
    })

    return list([file.filename for file in file_list])