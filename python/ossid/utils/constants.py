YCBV_OBJID2NAME = ['__background__', '002_master_chef_can', '003_cracker_box', '004_sugar_box', '005_tomato_soup_can', '006_mustard_bottle', \
                    '007_tuna_fish_can', '008_pudding_box', '009_gelatin_box', '010_potted_meat_can', '011_banana', '019_pitcher_base', \
                    '021_bleach_cleanser', '024_bowl', '025_mug', '035_power_drill', '036_wood_block', '037_scissors', '040_large_marker', \
                    '051_large_clamp', '052_extra_large_clamp', '061_foam_brick']

YCBV_OBJNAME2ID = {n:i for i, n in enumerate(YCBV_OBJID2NAME)}

SHAPENET_OBJECT_ID_OFFSET = 10000
BOP_OBJECT_ID_OFFSETS = {
    "hb": 100,
    "icbin": 200,
    "icmi": 300,
    "itodd": 400,
    "lm": 500,
    "lmo": 500,
    "ruapc": 700,
    "tless": 800,
    "tudl": 900,
    "tyol": 1000,
    "ycbv": 1100,
}