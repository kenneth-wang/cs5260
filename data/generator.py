from sklearn.model_selection import train_test_split
import pandas as pd
from typing import Generator, Tuple

COLS = ["ans_str", "query_str", "idx_of_ans", "idx_of_qns"]
INDEX_COL = "idx_of_ans"

TrainingSet, TestSet = pd.DataFrame, pd.DataFrame
TrainingOriSet = pd.DataFrame


def _generate_split(
    df_ori: pd.DataFrame,
    df_gen: pd.DataFrame,
    random_state: int,
) -> Tuple[TrainingSet, TrainingOriSet, TestSet]:
    """
    Args:
        df_ori: the original qn-ans pairs
        df_gen: the generated questions based on answerr
        random_state: some integer for reproducibility
    """
    assert df_ori.index.name == df_gen.index.name
    
    train_ori, test_ori = train_test_split(df_ori, test_size=0.25, random_state=random_state)
    commons: set = set(train_ori.index).intersection(set(df_gen.index))    
    train_gen = df_gen.loc[commons]
    
    train_aug = pd.concat([train_ori, train_gen])
    
    assert set(train_aug.index).intersection(set(test_ori.index)) == set()

    return train_aug, train_ori, test_ori


def generate_data(num_experiments: int) -> Generator:
    """
    Usage:
    
        >>> for train_set, test_set in generate_data(5):
        ...     model = train(train_set)
        ...     metric = eval(model(train_set), test_set)
    """
    
    df_ori = pd.read_csv(
        "data.csv", 
        index_col=INDEX_COL, 
        usecols=COLS,
    )
    df_gen = pd.read_csv(
        "generated_qns.csv", 
        index_col=INDEX_COL,
        usecols=COLS,
    )
    
    assert df_ori.shape[0] == len(df_ori.idx_of_qns.unique()) 

    for i in range(num_experiments):
        yield _generate_split(df_ori, df_gen, random_state=i)


if __name__ == "__main__":

    i = 1
    for x,y,z in generate_data(5):
        x.to_csv("train_"+str(i)+".csv")
        y.to_csv("trainorig_"+str(i)+".csv")
        z.to_csv("test_"+str(i)+".csv")

        i += 1