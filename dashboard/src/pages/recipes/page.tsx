
import { useLocation, useNavigate } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { LayoutGrid, Play, Zap, Loader2 } from 'lucide-react';
import { api } from '@/lib/api';

export default function RecipesPage() {
  const navigate = useNavigate();

  const { data, isLoading, error } = useQuery({
    queryKey: ['recipes'],
    queryFn: () => api.recipes.list(),
    staleTime: 60_000,
    retry: 2,
  });

  const recipes = data?.recipes || [];

  return (
    <div className="page-container">
      <div className="page-header">
        <h1>Recipe Catalog</h1>
        <p>Browse and launch declarative ML pipelines</p>
      </div>

      {isLoading && (
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, padding: 40, justifyContent: 'center', color: 'var(--text-dimmed)' }}>
          <Loader2 style={{ width: 20, height: 20, animation: 'spin 1s linear infinite' }} />
          Loading recipes...
        </div>
      )}

      {!isLoading && recipes.length === 0 && (
        <div style={{
          textAlign: 'center', padding: '60px 20px', color: 'var(--text-dimmed)',
          border: '1px dashed rgba(255,255,255,0.1)', borderRadius: 'var(--radius-lg)', marginTop: 20,
        }}>
          <LayoutGrid style={{ width: 32, height: 32, margin: '0 auto 12px', opacity: 0.4 }} />
          <div style={{ fontSize: 15, fontWeight: 600, marginBottom: 8 }}>No recipes found</div>
          <p style={{ fontSize: 13 }}>
            Add recipes to <code>projects/recipes/</code> to see them here.
            <br />
            Run <code>ml-plat recipe list</code> to verify available recipes.
          </p>
        </div>
      )}

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(380px, 1fr))', gap: 20 }}>
        {recipes.map((recipe: any) => (
          <div key={recipe.name} className="card" style={{ cursor: 'pointer' }} onClick={() => navigate(`/recipes/${recipe.name}`)}>
            <div className="card-body" style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                  <LayoutGrid style={{ width: 18, height: 18, color: 'var(--accent-primary)' }} />
                  <span style={{ fontWeight: 600, fontSize: 15 }}>{recipe.name}</span>
                  <span style={{ fontSize: 12, color: 'var(--text-dimmed)' }}>{recipe.version}</span>
                </div>
                {recipe.verified && (
                  <span style={{
                    fontSize: 10, fontWeight: 600, textTransform: 'uppercase',
                    letterSpacing: '0.06em', color: 'var(--success)',
                    background: 'rgba(16,185,129,0.12)', padding: '2px 8px', borderRadius: 100,
                  }}>
                    Verified
                  </span>
                )}
              </div>
              <p style={{ fontSize: 13, color: 'var(--text-muted)', lineHeight: 1.5 }}>{recipe.description}</p>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                {(recipe.tags || []).map((tag: string) => (
                  <span key={tag} style={{
                    fontSize: 11, fontWeight: 500, color: 'var(--accent-secondary)',
                    background: 'rgba(6,182,212,0.1)', padding: '2px 8px', borderRadius: 100,
                  }}>
                    {tag}
                  </span>
                ))}
              </div>
              <div style={{ display: 'flex', gap: 8, marginTop: 4 }}>
                <button className="btn btn-primary btn-sm" onClick={(e) => { e.stopPropagation(); navigate(`/recipes/${recipe.name}`); }}>
                  <Play style={{ width: 12, height: 12 }} /> Launch
                </button>
                <button className="btn btn-ghost btn-sm" onClick={(e) => { e.stopPropagation(); navigate(`/recipes/${recipe.name}`); }}>
                  <Zap style={{ width: 12, height: 12 }} /> Canary
                </button>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
