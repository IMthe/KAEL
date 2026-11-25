"""Phase-specific expressions selected by the knowledge-discovery agent.

Every coefficient, including the peak coordinates in cationic potential and Li-vacancy content,
is a free parameter. Roles only seed the random starting points of the optimiser.
"""
HCP_PHI = ("p0*exp(-p1*abs(phi-p2))", ["amplitude", "rate:phi", "center:phi"])
CCP_PHI = ("p0+p1*exp(-p2*abs(phi-p3))", ["offset", "amplitude", "rate:phi", "center:phi"])
HCP_PHI_NV = ("p0*exp(-p1*abs(phi-p2)-p3*abs(nv-p4))",
              ["amplitude", "rate:phi", "center:phi", "rate:nv", "center:nv"])
CCP_PHI_NV = ("p0+p1*exp(-p2*abs(phi-p3)-p4*abs(nv-p5))",
              ["offset", "amplitude", "rate:phi", "center:phi", "rate:nv", "center:nv"])
HCP_PHI_NV_CHI = ("p0*exp(-p1*abs(phi-p2)-p3*abs(nv-p4)+p5*chi)",
                  ["amplitude", "rate:phi", "center:phi", "rate:nv", "center:nv", "coefficient"])
CCP_PHI_NV_CHI = ("p0+p1*exp(-p2*abs(phi-p3)-p4*abs(nv-p5)+p6*chi)",
                  ["offset", "amplitude", "rate:phi", "center:phi", "rate:nv", "center:nv", "coefficient"])

# Literature model
LITERATURE = {"hcp": HCP_PHI_NV, "ccp": CCP_PHI_NV}

# Small-model descriptor tests
PANELS = {
    "a": ("literature", "vacancy_free", {"hcp": HCP_PHI, "ccp": CCP_PHI}),
    "b": ("literature", "all", {"hcp": HCP_PHI_NV, "ccp": CCP_PHI_NV}),
    "c": ("literature", "all", {"hcp": HCP_PHI_NV_CHI, "ccp": CCP_PHI_NV_CHI}),
    "d": ("combined", "vacancy_free", {"hcp": HCP_PHI, "ccp": CCP_PHI}),
    "e": ("combined", "all", {"hcp": HCP_PHI_NV, "ccp": CCP_PHI_NV}),
    "f": ("combined", "all", {"hcp": HCP_PHI_NV_CHI, "ccp": CCP_PHI_NV_CHI}),
}
