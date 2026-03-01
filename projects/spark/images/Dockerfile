FROM apache/spark:3.5.0

USER root

# Install Python ML dependencies
RUN pip install --no-cache-dir \
    pyspark==3.5.0 \
    pandas \
    numpy \
    pyarrow \
    boto3 \
    delta-spark==3.0.0

# Add Hadoop AWS connector for S3 access
ENV SPARK_HOME=/opt/spark
RUN wget -q -P ${SPARK_HOME}/jars/ \
    https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.3.4/hadoop-aws-3.3.4.jar && \
    wget -q -P ${SPARK_HOME}/jars/ \
    https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/1.12.262/aws-java-sdk-bundle-1.12.262.jar

WORKDIR /opt/spark/work-dir

USER ${spark_uid}
