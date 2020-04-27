import os
import gridfs
from pymongo import MongoClient

_client = MongoClient(os.environ.get('MONGO_URL', "mongodb://localhost:27017"))
_collection = _client.file_storage
_fs = gridfs.GridFS(_collection)
_collection.fs.files.create_index('uploadDate', expireAfterSeconds=10*24*60*60)


def write_file(job_id=None, file_identifier=None, file_path=None):
    if not job_id or not file_path or not file_identifier:
        raise Exception("Forgot to specify required parameter")

    with open(file_path, "rb") as f:
        index = _fs.put(f, filename=file_identifier, job=job_id)

    return index


def get_file(job_id=None, file_identifier=None):
    if not job_id or not file_identifier:
        raise Exception("Forgot to specify required parameter")

    db_file = _fs.find_one(filter={
        "filename": file_identifier,
        "job": job_id
    })

    if db_file:
        return _fs.get(db_file._id)
    else:
        return None
