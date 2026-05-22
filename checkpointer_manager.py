from langgraph.checkpoint.postgres import PostgresSaver




from langgraph.checkpoint.serde.encrypted import EncryptedSerializer
from langgraph.checkpoint.postgres import PostgresSaver
import psycopg
from psycopg.rows import dict_row



conn = psycopg.connect(
    "postgres://postgres:Aa123456@localhost:5432/graphmem?sslmode=disable",
    autocommit=True, # Critical for setup()
    row_factory=dict_row
)


checkpointer = PostgresSaver(conn)

config = {"configurable": {"thread_id": "default_user"}}



        
        