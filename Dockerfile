# Use the Miniconda3 base image
FROM continuumio/miniconda3

# Set the working directory in the container
WORKDIR /LLM_ANALYSIS

# Copy the environment.yml file into the container
COPY environment.yml .
COPY .  /LLM_ANALYSIS/
# Create the environment from the yml file
RUN conda env create -f environment.yml

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "analyst", "/bin/bash", "-c"]

# Copy the rest of your application code

# Set the entrypoint to your script
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "analyst", "python"]
