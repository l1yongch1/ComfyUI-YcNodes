from .nodes import *

nodes_config = {
     # nodes
     "RemoveHighlightAndBlur": {'class':RemoveHighlightAndBlur, 'name':'RemoveHighlightAndBlur'}
    ,"RoundedCorners":{'class':RoundedCorners, 'name':'RoundedCorners'}
    ,"PaddingAccordingToBackground":{'class':PaddingAccordingToBackground, 'name':'PaddingAccordingToBackground'}
    ,"QwenCaption":{'class':QwenCaption,'name':'QwenCaption'}
    ,"RemoveBackground":{'class':RemoveBackground,'name':'RemoveBackground'}
    ,'RemoveBackgroundWithProtection':{'class':RemoveBackgroundWithProtection,'name':'RemoveBackgroundWithProtection'}
    ,'RemoveBackgroundWithProtectionOptimized':{'class':RemoveBackgroundWithProtectionOptimized, 'name':'RemoveBackgroundWithProtectionOptimized'}
    ,"EstimateBackgroundFromTriangleCorners":{'class':EstimateBackgroundFromTriangleCorners,'name':'EstimateBackgroundFromTriangleCorners'}
}

def analysis_nodes_config(nodes_config):
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

    for node_name,node_info in nodes_config.items():
        NODE_CLASS_MAPPINGS[node_name] = node_info['class']
        NODE_DISPLAY_NAME_MAPPINGS[node_name] = node_info['name']

    return NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# __all__ = [NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS]
