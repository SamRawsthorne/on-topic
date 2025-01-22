from lda_eval_lib.gensim_ldamallet import LdaMallet
import os

class CustomLdaMallet(LdaMallet):
    """
    Custom LdaMallet class with support for additional options, such as 'beta'.
    """
    def __init__(self, mallet_path, corpus, num_topics=100, alpha='symmetric', beta=None, id2word=None, iterations=1000, workers=4):
        if beta == 'auto':
            beta = 0.01  # Mallet uses 0.01 beta unless a specific value is specified
        self.beta = beta  # Store the beta parameter
        super().__init__(
            mallet_path=mallet_path,
            corpus=corpus,
            num_topics=num_topics,
            alpha=alpha,
            id2word=id2word,
            iterations=iterations
        )
    
    def _create_mallet_command(self):
        """
        Override the command to include beta if specified.
        """
        command = super()._create_mallet_command()
        if self.beta is not None:
            command += f" --beta {self.beta}"  # Append the beta option
        return command
