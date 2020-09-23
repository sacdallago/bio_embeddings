# The bio_embeddings webserver

The webserver provides an easy-to-use web interface to a part of the functionality of bio_embeddings. You can run the
 webserver with docker. You need
 
 * A mongodb container 
 * A webserver container
 * A worker container (celery)
 * An ampq broker (rabbitmq)
  
The worker should run on a host with a GPU.

