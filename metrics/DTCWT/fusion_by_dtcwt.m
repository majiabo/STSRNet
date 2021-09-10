function y = fusion_by_dtcwt(x1 , x2)
y = zeros(size(x1));
y_r = m_dtcwt(x1(: ,: , 1),x2(: , : , 1));
y_g = m_dtcwt(x1(: ,: , 2),x2(: , : , 2));
y_b = m_dtcwt(x1(: ,: , 3),x2(: , : , 3));
y(: , : , 1) = y_r;
y(: , : , 2) = y_g;
y(: , : , 3) = y_b;
end