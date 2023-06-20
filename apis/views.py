from rest_framework.decorators import api_view
from rest_framework.response import Response


@api_view(["GET"])
def index(request):
    resp = {
        "endpoints": [
            "GET /api/prediction to browse the predict form",
            "POST /api/prediction to submit prediction",
        ],
    }
    return Response(resp)


@api_view(["GET", "POST"])
def categorize_abstract(request):
    abstract = request.data
    # predict with model

    return Response()
