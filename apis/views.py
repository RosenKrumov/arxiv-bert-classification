from rest_framework.decorators import api_view
from rest_framework.response import Response

from arxiv_bert.model.predict import ArxivBertPredictor

arxiv_predictor = ArxivBertPredictor()


@api_view(["GET"])
def index(request):
    resp = {
        "endpoints": [
            "GET /api/prediction to browse the predict form",
            "POST /api/prediction to submit prediction",
        ],
    }
    return Response(resp)


@api_view(["POST"])
def categorize_abstract(request):
    abstract = request.data
    if "text" not in abstract:
        return Response(
            {"error": "Empty input"},
            status=400,
        )

    prediction = arxiv_predictor.predict(abstract["text"])
    return Response({"prediction": prediction})
