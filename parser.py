import pandas as pd

HEADERS = 'GDEKT	GDENR	GDENAMK	EGID	GPARZ	GGBKR	GEBNR	GBEZ	ESTRID	STRNAMK1	DEINR	PLZ4	PLZZ	PLZNAMK	GKODX	GKODY	GSTAT	GBAUJ	GBAUP	GKAT	GKLAS	GAREA	GASTW	GANZWHG	GAZZI	GHEIZ	GENHZ	GWWV	GENWW	DMUTDAT	DEXPDAT'.split('\t')
COL_TO_KEEP = ['EGID', 'heatingNeeds(Wh)', 'coolingNeeds(Wh)', 'GBAUP', 'GASTW', 'GAREA', 'GKLAS']

if __name__ == '__main__':
    output = pd.read_csv('data/jonctionnofloors_YearlyResultsPerBuilding.out', delimiter='\t')
    output['EGID'] = output['#buildingId(key)'].apply(lambda s: int(s[s.find('(') + 1: s.find(')')]))

    info = pd.read_csv('data/EPFL-150701-GEB.txt', delimiter='\t', names=HEADERS, header=None, encoding='iso-8859-1')
    info.drop_duplicates(subset='EGID', inplace=True)
    joined = output.join(info.set_index('EGID'), on='EGID', lsuffix='_', how='left')[COL_TO_KEEP]

    joined.to_csv('data/sanitized.csv', index=False)
