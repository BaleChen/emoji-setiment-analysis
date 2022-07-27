class TweetTrainingExample:
    """Structure holding a Tweet Training example"""

    def __init__(self, id, text, label):
        """Create the training example
        Args:
            id: ID of the example
            text: text of the example
            label: example label
        """
        self.id = id
        self.text = text
        self.label = label

    def __repr__(self):
        return str.format('{}, {}, {}\n', self.id, self.label, self.text)
  
if __name__ == "__main__":
  example_data = pd.read_pickle('emoji2vec data/examples.p')
  train_data = pd.read_pickle('emoji2vec data/train.p')
  test_data = pd.read_pickle('emoji2vec data/test.p')
  
  # Load pickle files into xlsx
  ID, label, content = [],[],[]
  for i in train_data:
      ID.append(i.id)
      label.append(i.label)
      content.append(i.text)

  train = pd.DataFrame({'ID': ID, 'label':label, 'content':content})
  train.to_excel('emoji2vec_train.xlsx', index=False)
  
  # Load pickle files into xlsx
  ID, label, content = [],[],[]
  for i in test_data:
      ID.append(i.id)
      label.append(i.label)
      content.append(i.text)

  train = pd.DataFrame({'ID': ID, 'label':label, 'content':content})
  train.to_excel('emoji2vec_test.xlsx', index=False)
