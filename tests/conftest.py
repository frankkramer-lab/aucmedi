import warnings

# Suppress noisy DeprecationWarnings from third-party packages (scipy.ndimage.filters)
warnings.filterwarnings("ignore", category=DeprecationWarning)
