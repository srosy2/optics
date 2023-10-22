# Use the official PostgreSQL image as the base image
FROM postgres

# Set the environment variables
ENV POSTGRES_PASSWORD=123

# Create and switch to a working directory
WORKDIR /app

# Copy a SQL script with the necessary commands
COPY init.sql /docker-entrypoint-initdb.d/

# Expose the PostgreSQL default port
EXPOSE 5432

# Add a named volume for data persistence
VOLUME /var/lib/postgresql/data

# Start the PostgreSQL server
CMD ["postgres"]

# Build and run this Dockerfile as follows:
# docker build -t my-postgres-container .
# docker run -d --name postgres-container -p 5432:5432 -v postgres-data:/var/lib/postgresql/data my-postgres-container