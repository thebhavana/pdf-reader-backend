from rest_framework import serializers

class UploadSerializer(serializers.Serializer):
    file = serializers.FileField()

class QuerySerializer(serializers.Serializer):
    question = serializers.CharField()
    file_path = serializers.CharField(required=False, allow_blank=True)
