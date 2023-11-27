from yarp.input_parsers import xyz_parse

E, G=xyz_parse("/home/hsu205/classy-yarp/reaction/reaction_xyz/0-0.xyz", multiple=True)
print(E[0])
print(G[0])
