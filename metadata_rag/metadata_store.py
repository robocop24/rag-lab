class MetadataStore:
    
    def __init__(self, chunks):
        
        self.chunks = chunks
        
    def filter(self, project=None, module=None):
        
        results = self.chunks
        
        if project:
            results = [
                chunk for chunk in results
                if chunk["project"] == project
            ]
            
        if module:
            results = [
                chunk for chunk in results
                if chunk["project"] == project
            ]           
            
        return results