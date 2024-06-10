FROM spark:3.5.1-scala2.12-java11-ubuntu
ARG SCALA_VERSION=2.12
ARG DELTA_VERSION=3.2.0

USER root

ENV SPARK_HOME=/opt/spark

ADD https://repo1.maven.org/maven2/io/delta/delta-spark_$SCALA_VERSION/$DELTA_VERSION/delta-spark_$SCALA_VERSION-$DELTA_VERSION.jar ${SPARK_HOME}/jars/delta-spark_$SCALA_VERSION-$DELTA_VERSION.jar
ADD https://repo1.maven.org/maven2/io/delta/delta-storage/$DELTA_VERSION/delta-storage-$DELTA_VERSION.jar ${SPARK_HOME}/jars/delta-storage-$DELTA_VERSION.jar
ADD https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.3.4/hadoop-aws-3.3.4.jar $SPARK_HOME/jars/hadoop-aws-3.3.4.jar
ADD https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/1.12.367/aws-java-sdk-bundle-1.12.367.jar $SPARK_HOME/jars/aws-java-sdk-bundle-1.12.367.jar

RUN chown -R spark:spark $SPARK_HOME/jars/

USER spark

ENTRYPOINT [ "/opt/entrypoint.sh" ]