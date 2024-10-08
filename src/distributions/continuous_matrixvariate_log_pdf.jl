export ContinuousMatrixvariateLogPdf


struct ContinuousMatrixvariateLogPdf{T <: Tuple, F} <: AbstractContinuousGenericLogPdf
    domain::T
    logpdf::F
end

dimension(d::Tuple) = dimension.(d)
in(x::AbstractMatrix, domain::Tuple) = (size(x) == dimension(domain))
insupport(d::ContinuousMatrixvariateLogPdf, x) = true  # TODO: make right
