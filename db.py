import clickhouse_connect

# Initialize the ClickHouse client
client = clickhouse_connect.get_client(
    host='host name',
    port=port,
    username='userid',
    password='pasword'
)

print("Connected to ClickHouse!")

result = client.command("SELECT version()")
print("ClickHouse version:", result)
