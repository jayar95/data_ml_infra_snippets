FROM apache/kyuubi:1.9.0
ARG SCALA_VERSION=2.12
ARG DELTA_VERSION=3.2.0


USER root
ENV SPARK_HOME=/opt/spark

ADD https://dlcdn.apache.org/spark/spark-3.5.1/spark-3.5.1-bin-hadoop3.tgz spark-3.5.1-bin-hadoop3.tgz
RUN tar zxvf spark-3.5.1-bin-hadoop3.tgz \
  && mv spark-3.5.1-bin-hadoop3/ /opt/spark \
  && rm spark-3.5.1-bin-hadoop3.tgz \
  && rm -rf spark-3.5.1-bin-hadoop3/

ADD https://repo1.maven.org/maven2/io/delta/delta-spark_$SCALA_VERSION/$DELTA_VERSION/delta-spark_$SCALA_VERSION-$DELTA_VERSION.jar ${SPARK_HOME}/jars/delta-spark_$SCALA_VERSION-$DELTA_VERSION.jar
ADD https://repo1.maven.org/maven2/io/delta/delta-storage/$DELTA_VERSION/delta-storage-$DELTA_VERSION.jar ${SPARK_HOME}/jars/delta-storage-$DELTA_VERSION.jar
ADD https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.3.4/hadoop-aws-3.3.4.jar $SPARK_HOME/jars/hadoop-aws-3.3.4.jar
ADD https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/1.12.262/aws-java-sdk-bundle-1.12.262.jar $SPARK_HOME/jars/aws-java-sdk-bundle-1.12.262.jar
ADD https://repo1.maven.org/maven2/org/apache/spark/spark-hadoop-cloud_2.12/3.5.1/spark-hadoop-cloud_2.12-3.5.1.jar $SPARK_HOME/jars/spark-hadoop-cloud_2.12-3.5.1.jar

ADD https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.3.6/hadoop-aws-3.3.6.jar /opt/kyuubi/jars/hadoop-aws-3.3.6.jar
ADD https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/1.12.367/aws-java-sdk-bundle-1.12.367.jar /opt/kyuubi/jars/aws-java-sdk-bundle-1.12.367.jar

RUN chmod 644 $SPARK_HOME/jars/hadoop-aws-3.3.4.jar \
    && chmod 644 $SPARK_HOME/jars/aws-java-sdk-bundle-1.12.262.jar \
    && chmod 644 $SPARK_HOME/jars/spark-hadoop-cloud_2.12-3.5.1.jar \
    && chmod 644 ${SPARK_HOME}/jars/delta-storage-$DELTA_VERSION.jar \
    && chmod 644 ${SPARK_HOME}/jars/delta-spark_$SCALA_VERSION-$DELTA_VERSION.jar \
    && chmod 644 /opt/kyuubi/jars/hadoop-aws-3.3.6.jar \
    && chmod 644 /opt/kyuubi/jars/aws-java-sdk-bundle-1.12.367.jar

RUN chown -R kyuubi /opt/spark

USER kyuubi