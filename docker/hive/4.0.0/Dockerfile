FROM apache/hive:4.0.0

USER root

ADD https://repo1.maven.org/maven2/org/postgresql/postgresql/42.7.3/postgresql-42.7.3.jar /opt/hive/lib/postgres.jar
ADD https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.3.6/hadoop-aws-3.3.6.jar /opt/hive/lib/hadoop-aws-3.3.6.jar
ADD https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/1.12.367/aws-java-sdk-bundle-1.12.367.jar /opt/hive/lib/aws-java-sdk-bundle-1.12.367.jar

RUN chmod 644 /opt/hive/lib/postgres.jar \
    && chmod 644 /opt/hive/lib/hadoop-aws-3.3.6.jar \
    && chmod 644 /opt/hive/lib/aws-java-sdk-bundle-1.12.367.jar

RUN chown -R hive:hive /opt/hive/lib

USER hive