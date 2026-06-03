# Minimal frontend init for inference only
try:
    from .w2vbert import W2VBertFrontend
except ImportError:
    W2VBertFrontend = None

frontend_class_dict = {
    'fbank': None,
    'w2vbert': W2VBertFrontend,
}
